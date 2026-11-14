# -*- coding: utf-8 -*-
"""
“识别与提取分开跑会不会更快”实测探针。

分三轮，同一段视频、同一批帧、同一套模型：
  A. 纯解码：只 cap.read()，量 4K 解码到底占多少
  B. 串行（现状）：解码 → det → rec 全在一个线程里顺序跑
  C. 流水线（分离）：解码线程 + det 线程 + rec 线程，用队列串联

对比 A/B/C 的墙钟时间，就能判断“分开”值不值得、上线收益多少。

用法：python profile_split.py [视频路径] [起始秒] [帧数]
"""
import os
import sys
import time
import queue
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import cv2
# 可选：限制 OpenCV 内部线程数，验证多线程流水线下的超订问题
_cv_threads = os.environ.get('VSE_CV_THREADS')
if _cv_threads:
    cv2.setNumThreads(int(_cv_threads))
    print(f'cv2.setNumThreads({_cv_threads})')
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.ocr import OcrRecogniser

VIDEO = sys.argv[1] if len(sys.argv) > 1 else 'railgun_2min.mp4'
START_SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
N_FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 300
STRIP = (0.78, 0.99, 0.05, 0.95)   # ymin,ymax,xmin,xmax

ha = HardwareAccelerator.instance()
ha.initialize()
print(f'视频 {VIDEO} 起始 {START_SEC}s 帧数 {N_FRAMES}')

print('初始化模型...')
t0 = time.perf_counter()
det = SubtitleDetect()
ocr = OcrRecogniser()
ocr.hardware_accelerator = ha
ocr.recogniser = ocr.init_model()
rec_model = ocr._get_rec_model()
print(f'模型加载 {time.perf_counter() - t0:.1f}s  rec子模型={type(rec_model).__name__ if rec_model else None}')

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
y0, y1 = int(STRIP[0] * h), int(STRIP[1] * h)
x0, x1 = int(STRIP[2] * w), int(STRIP[3] * w)
print(f'{w}x{h} @{fps:.2f}fps  字幕条 y[{y0}:{y1}] x[{x0}:{x1}]')

start_frame = int(START_SEC * fps)


def seek_cap():
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)


# ---------------- 预热 ----------------
seek_cap()
ret, frame = cap.read()
strip = frame[y0:y1, x0:x1]
for _ in range(3):
    det.detect_subtitle(strip)
warm_polys, _ = det.detect_subtitle(strip)
if len(warm_polys):
    for _ in range(2):
        ocr.recognise_polys(frame, warm_polys)
print('预热完成\n')

# ---------------- A. 纯解码 ----------------
seek_cap()
t0 = time.perf_counter()
for _ in range(N_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
t_decode_only = time.perf_counter() - t0
print(f'A. 纯解码          : {t_decode_only:6.2f}s  {t_decode_only / N_FRAMES * 1000:6.2f} ms/帧')

# ---------------- B. 串行（现状） ----------------
seek_cap()
n_text = 0
t_read = t_det = t_rec = 0.0
t0 = time.perf_counter()
for _ in range(N_FRAMES):
    t = time.perf_counter()
    ret, frame = cap.read()
    t_read += time.perf_counter() - t
    if not ret:
        break
    t = time.perf_counter()
    polys, _ = det.detect_subtitle(frame[y0:y1, x0:x1])
    t_det += time.perf_counter() - t
    if len(polys):
        t = time.perf_counter()
        ocr.recognise_polys(frame, polys)
        t_rec += time.perf_counter() - t
        n_text += 1
t_serial = time.perf_counter() - t0
print(f'B. 串行（现状）    : {t_serial:6.2f}s  {t_serial / N_FRAMES * 1000:6.2f} ms/帧'
      f'   [读取 {t_read / N_FRAMES * 1000:.1f} + det {t_det / N_FRAMES * 1000:.1f}'
      f' + rec {t_rec / N_FRAMES * 1000:.1f} ms/帧，含字幕 {n_text}/{N_FRAMES} 帧]')

# ---------------- C. 流水线（提取与识别分离） ----------------
def run_pipeline(n_frames):
    q_strip = queue.Queue(maxsize=8)     # (idx, strip)
    q_crops = queue.Queue(maxsize=16)    # (idx, crops)
    stats = {'read': 0.0, 'det': 0.0, 'rec': 0.0, 'text': 0, 'frames': 0}
    lock = threading.Lock()
    stop = threading.Event()

    def producer():     # 解码 + 裁剪（相当于“提取”）
        seek_cap()
        for i in range(n_frames):
            t = time.perf_counter()
            ret, fr = cap.read()
            dt = time.perf_counter() - t
            if not ret:
                break
            s = fr[y0:y1, x0:x1].copy()   # copy：脱离原帧，模拟独立数据
            with lock:
                stats['read'] += dt
                stats['frames'] += 1
            q_strip.put((i, s))
        q_strip.put(None)

    def det_worker():
        while True:
            item = q_strip.get()
            if item is None:
                q_crops.put(None)
                return
            i, s = item
            t = time.perf_counter()
            polys, _ = det.detect_subtitle(s)
            dt = time.perf_counter() - t
            with lock:
                stats['det'] += dt
            if len(polys):
                crops, keep = OcrRecogniser._extract_crops(s, polys)
                with lock:
                    stats['text'] += 1
                q_crops.put((i, crops))
            else:
                q_crops.put((i, None))

    def rec_worker():
        while True:
            item = q_crops.get()
            if item is None:
                return
            i, crops = item
            if not crops:
                continue
            t = time.perf_counter()
            list(rec_model(crops))
            dt = time.perf_counter() - t
            with lock:
                stats['rec'] += dt

    ts = [threading.Thread(target=producer, daemon=True),
          threading.Thread(target=det_worker, daemon=True),
          threading.Thread(target=rec_worker, daemon=True)]
    t0 = time.perf_counter()
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    return time.perf_counter() - t0, stats


t_pipe, st = run_pipeline(N_FRAMES)
print(f'C. 流水线（分离）  : {t_pipe:6.2f}s  {t_pipe / N_FRAMES * 1000:6.2f} ms/帧'
      f'   [读取 {st["read"] / N_FRAMES * 1000:.1f} + det {st["det"] / N_FRAMES * 1000:.1f}'
      f' + rec {st["rec"] / N_FRAMES * 1000:.1f} ms/帧，含字幕 {st["text"]}/{N_FRAMES} 帧]')

print()
print(f'分离 vs 串行: {t_serial / t_pipe:.2f}x   节省 {t_serial - t_pipe:.1f}s / {N_FRAMES} 帧')
print(f'理论下限（只看最慢环节）: {max(st["read"], st["det"] + st["read"], st["rec"]):.2f}s')

cap.release()
