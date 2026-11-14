# -*- coding: utf-8 -*-
"""补充探针：det 的 predictor 层 batch 测试 + 纯调用开销下限"""
import sys, os, time
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.subtitle_detect import SubtitleDetect

ha = HardwareAccelerator.instance(); ha.initialize()
det = SubtitleDetect()
td = det.text_detector

# 构造测试输入：真实字幕条 + 空白条
import imageio_ffmpeg
VIDEO = "//192.168.5.59/动漫/某科学的超电磁炮/某科学的超电磁炮S01E01 - 电击使 Electro Master.mp4"
meta = next(imageio_ffmpeg.read_frames(VIDEO))
it = imageio_ffmpeg.read_frames(VIDEO)
next(it)
h, w = meta["size"][1], meta["size"][0]
strips = []
for i, f in enumerate(it):
    if i < 7200: continue
    fr = np.frombuffer(f, np.uint8).reshape(h, w, 3)
    strips.append(fr[1684:2138, 192:3648].copy())
    if len(strips) >= 16: break
print("字幕条:", strips[0].shape)

def bench(fn, n=20):
    fn(); fn()  # warmup
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return sum(ts)/n*1000

print("\n=== det batch（predictor 层，16 条真实字幕条）===")
t1 = bench(lambda: [td.predict(s) for s in strips])
print(f"单条循环: {t1:.2f} ms/条")
t2 = bench(lambda: list(td.predict(strips)))
print(f"一次batch: {t2:.2f} ms/条  (加速 {t1/t2:.2f}x)")

blank = np.zeros((454, 3456, 3), np.uint8)
t3 = bench(lambda: td.predict(blank))
print(f"空白条单张: {t3:.2f} ms  <- det 纯调用开销下限")

# batch 大小扫描
print("\n=== rec/det batch 大小扫描（ms/条）===")
crops = []
for fr in []: pass
# rec crops：从 strips 里随便抠一块当 crop 太假，直接用空白小图 + 真实条都试 det
for bs in (4, 8, 16):
    t = bench(lambda: list(td.predict(strips[:bs])))
    print(f"det batch={bs}: {t/bs:.2f} ms/条")
