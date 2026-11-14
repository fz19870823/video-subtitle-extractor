# -*- coding: utf-8 -*-
"""
瓶颈定位探针：把精确模式每帧耗时拆到 解码 / det / rec，并实测 GPU 加速可行性。

测试项：
  A. 分段计时（150 帧）：解码、字幕条裁剪、det、rec、后处理
  B. det/rec 尺寸缩放测试：输入缩小一半若耗时不变 => 固定开销主导（GPU 算力不是瓶颈）
  C. predictor 层 batch 测试：单张循环 vs 一次 batch（验证 GPU batch 是否有收益）
  D. TensorRT 可用性检查
  E. 后台采样 GPU 利用率（nvidia-smi）
"""
import sys
import os
import time
import threading
import subprocess
from statistics import mean, median

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
from backend.config import config
from backend.tools.hardware_accelerator import HardwareAccelerator

VIDEO = "//192.168.5.59/动漫/某科学的超电磁炮/某科学的超电磁炮S01E01 - 电击使 Electro Master.mp4"
SKIP_SECONDS = 300      # 跳到 5 分钟处（正片有字幕）
N_FRAMES = 150
STRIP = (0.78, 0.99, 0.05, 0.95)  # ymin,ymax,xmin,xmax，与 config.json 一致

ha = HardwareAccelerator.instance()
ha.initialize()
print("=== 环境 ===")
print("has_cuda:", ha.has_cuda())
try:
    import paddle
    print("paddle:", paddle.__version__, "| compiled with cuda:", paddle.device.is_compiled_with_cuda())
except Exception as e:
    print("paddle import fail:", e)

# GPU 利用率采样线程
gpu_samples = []
sampling = [True]
def sample_gpu():
    while sampling[0]:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            vals = out.stdout.strip().split(",")
            gpu_samples.append(int(vals[0]))
        except Exception:
            pass
        time.sleep(0.2)

t_sampler = threading.Thread(target=sample_gpu, daemon=True)
t_sampler.start()

from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.ocr import OcrRecogniser

print("初始化模型...")
t0 = time.perf_counter()
det = SubtitleDetect()
ocr = OcrRecogniser()
ocr.hardware_accelerator = ha
ocr.recogniser = ocr.init_model()
print(f"模型加载 {time.perf_counter()-t0:.1f}s")

rec_model = ocr._get_rec_model()
print("rec 子模型:", type(rec_model).__name__ if rec_model else None)

# ---------------- 读帧 ----------------
import imageio_ffmpeg
print(f"读取视频（跳过前 {SKIP_SECONDS}s）...")
t0 = time.perf_counter()
frames_iter = imageio_ffmpeg.read_frames(VIDEO)
meta = next(frames_iter)
frames = []
for i, f in enumerate(frames_iter):
    frame = np.frombuffer(f, np.uint8).reshape(meta["size"][1], meta["size"][0], 3)
    if i >= SKIP_SECONDS * 24:  # 近似按 24fps 跳帧
        frames.append(frame.copy())
        if len(frames) >= N_FRAMES:
            break
print(f"取到 {len(frames)} 帧 ({time.perf_counter()-t0:.1f}s)")

h, w = frames[0].shape[:2]
y0, y1 = int(STRIP[0] * h), int(STRIP[1] * h)
x0, x1 = int(STRIP[2] * w), int(STRIP[3] * w)
print(f"分辨率 {w}x{h}，字幕条 y[{y0}:{y1}] x[{x0}:{x1}]")

# ---------------- 预热 ----------------
strip = frames[0][y0:y1, x0:x1]
for _ in range(3):
    det.detect_subtitle(strip)
crops_warm = []
for fr in frames[:5]:
    polys, _ = det.detect_subtitle(fr[y0:y1, x0:x1])
    if len(polys):
        crops_warm.append(fr)
for fr in crops_warm[:2]:
    polys, _ = det.detect_subtitle(fr[y0:y1, x0:x1])
    ocr.recognise_polys(fr, polys)
print("预热完成\n")

# ---------------- A. 分段计时 ----------------
print("=== A. 逐帧分段计时 ===")
t_decode = t_crop = t_det = t_rec = t_post = 0.0
n_with_text = 0
all_polys = []
per_frame = []
frames_iter = imageio_ffmpeg.read_frames(VIDEO)
meta = next(frames_iter)
idx = 0
skip_target = SKIP_SECONDS * 24
for f in frames_iter:
    if idx < skip_target:
        idx += 1
        continue
    t = time.perf_counter()
    frame = np.frombuffer(f, np.uint8).reshape(meta["size"][1], meta["size"][0], 3)
    t_decode += time.perf_counter() - t

    t = time.perf_counter()
    strip = frame[y0:y1, x0:x1]
    t_crop += time.perf_counter() - t

    t = time.perf_counter()
    polys, _ = det.detect_subtitle(strip)
    t_det += time.perf_counter() - t

    t = time.perf_counter()
    if len(polys):
        dt_box, rec_res = ocr.recognise_polys(frame, polys)
        n_with_text += 1
    t_rec += time.perf_counter() - t
    all_polys.append(polys)
    per_frame.append(0)
    idx += 1
    if idx - skip_target >= N_FRAMES:
        break

sampling[0] = False
n = len(per_frame)
tot = t_decode + t_crop + t_det + t_rec + t_post
print(f"帧数 {n}，含字幕帧 {n_with_text}")
print(f"解码  : {t_decode/n*1000:7.2f} ms/帧  ({t_decode/tot:.0%})")
print(f"裁剪  : {t_crop/n*1000:7.2f} ms/帧  ({t_crop/tot:.0%})")
print(f"det   : {t_det/n*1000:7.2f} ms/帧  ({t_det/tot:.0%})")
print(f"rec   : {t_rec/n*1000:7.2f} ms/帧  ({t_rec/tot:.0%})")
print(f"其他  : {t_post/n*1000:7.2f} ms/帧  ({t_post/tot:.0%})")
print(f"合计  : {tot/n*1000:7.2f} ms/帧  ({1000/tot*n if tot else 0:.1f} fps)")
if gpu_samples:
    print(f"GPU利用率: mean={mean(gpu_samples):.0f}% median={median(gpu_samples):.0f}% max={max(gpu_samples)}%  样本数={len(gpu_samples)}")

# ---------------- B. 尺寸缩放测试 ----------------
print("\n=== B. 输入尺寸缩放 => 是否固定开销主导 ===")
def bench(fn, n=30):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return mean(ts) * 1000

strip_full = frames[0][y0:y1, x0:x1]
strip_half = strip_full[::2, ::2]
strip_quarter = strip_full[::4, ::4]
print(f"det 100%尺寸({strip_full.shape[1]}x{strip_full.shape[0]}): {bench(lambda: det.detect_subtitle(strip_full)):.2f} ms")
print(f"det  50%尺寸({strip_half.shape[1]}x{strip_half.shape[0]}): {bench(lambda: det.detect_subtitle(strip_half)):.2f} ms")
print(f"det  25%尺寸({strip_quarter.shape[1]}x{strip_quarter.shape[0]}): {bench(lambda: det.detect_subtitle(strip_quarter)):.2f} ms")

# rec 用一个真实 crop
crop = None
for fr, polys in zip(frames, all_polys):
    if len(polys):
        p = polys[0]
        xs, ys = p[:, 0], p[:, 1]
        crop = fr[int(ys.min()):int(ys.max()), int(xs.min()):int(xs.max())]
        break
if crop is not None:
    ch, cw = crop.shape[:2]
    crop_half = crop[::2, ::2]
    rec_m = ocr._get_rec_model()
    print(f"rec 100%尺寸({cw}x{ch}): {bench(lambda: list(rec_model([crop]))):.2f} ms")
    print(f"rec  50%尺寸({crop_half.shape[1]}x{crop_half.shape[0]}): {bench(lambda: list(rec_model([crop_half]))):.2f} ms")
    blank = np.zeros((48, 320, 3), np.uint8)
    print(f"rec 空白图(320x48)  : {bench(lambda: list(rec_model([blank]))):.2f} ms  <- 纯调用开销下限")

# ---------------- C. predictor 层 batch 测试 ----------------
print("\n=== C. predictor 层 batch：单张循环 vs 一次 batch ===")
batch_crops = []
for fr, polys in zip(frames, all_polys):
    if len(polys):
        p = polys[0]
        xs, ys = p[:, 0], p[:, 1]
        c = fr[int(ys.min()):int(ys.max()), int(xs.min()):int(xs.max())]
        if c.size:
            batch_crops.append(c)
    if len(batch_crops) >= 32:
        break
print(f"收集到 {len(batch_crops)} 个 crop")
if len(batch_crops) >= 8:
    t = time.perf_counter()
    for c in batch_crops:
        list(rec_model([c]))
    t_single = (time.perf_counter() - t) / len(batch_crops) * 1000
    t = time.perf_counter()
    list(rec_model(batch_crops))
    t_batch = (time.perf_counter() - t) / len(batch_crops) * 1000
    print(f"单张循环: {t_single:.2f} ms/crop")
    print(f"一次batch: {t_batch:.2f} ms/crop  (加速 {t_single/t_batch:.2f}x)")

# ---------------- D. TensorRT 可用性 ----------------
print("\n=== D. TensorRT 可用性 ===")
try:
    from paddle.inference import Config
    print("paddle.inference.Config 存在；当前 CUDA 环境是否内置 TRT 取决于 wheel，")
    print("PaddleOCR 3.x 传参方式：PaddleOCR(..., enable_trt=True) 或 pp_option，需实测。")
    try:
        import paddle.tensorrt  # noqa
        print("paddle.tensorrt 可导入")
    except ImportError:
        print("paddle.tensorrt 不可导入（该 wheel 未打包 TRT 支持）")
except Exception as e:
    print("检查失败:", e)

print("\n完成")
