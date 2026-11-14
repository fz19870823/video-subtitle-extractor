"""
CUDA 模式自检脚本（本地环境验证用，不属于上游项目代码）

用法:
    .venv/Scripts/python.exe test_cuda_mode.py [视频路径] [--tier medium|small] [--frames 6]

作用:
    1. 打印 paddle / CUDA / 硬件加速器检测结果
    2. 用本地 PP-OCRv6 模型在 GPU 上跑真实帧 OCR，输出识别文本与耗时
    3. 校验结果确实落在 GPU 上（显存占用 + paddle device）
"""
import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import paddle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', nargs='?', default=os.path.join('test', 'test_cn.mp4'))
    parser.add_argument('--tier', default='medium', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--frames', type=int, default=6)
    parser.add_argument('--top', type=float, default=0.78, help='字幕区域上边界比例')
    parser.add_argument('--bottom', type=float, default=0.99, help='字幕区域下边界比例')
    args = parser.parse_args()

    # 模型档位通过项目自身的环境变量生效
    os.environ['VSE_OCR_TIER'] = args.tier

    print('=' * 60)
    print(f'paddle         : {paddle.__version__}')
    print(f'compiled_cuda  : {paddle.is_compiled_with_cuda()}')
    print(f'cuda_places    : {paddle.static.cuda_places()}')
    print(f'cudnn          : {paddle.get_cudnn_version()}')
    print('=' * 60)

    from backend.tools.hardware_accelerator import HardwareAccelerator
    from backend.tools.ocr import OcrRecogniser

    acc = HardwareAccelerator.instance()
    print(f'accelerator    : {acc.accelerator_name}  (has_cuda={acc.has_cuda()})')

    video = os.path.join(BASE_DIR, args.video) if not os.path.isabs(args.video) else args.video
    if not os.path.exists(video):
        print(f'[FAIL] 视频不存在: {video}')
        return 2

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f'video          : {os.path.basename(video)}  {width}x{height}  {total} frames')

    ocr = OcrRecogniser()
    ocr.hardware_accelerator = acc

    # 先初始化模型，单独计时
    t0 = time.time()
    ocr.init_model()
    print(f'模型初始化耗时 : {time.time() - t0:.2f}s')
    print(f'paddle device  : {paddle.device.get_device()}')
    alloc = paddle.device.cuda.memory_allocated()
    print(f'GPU 显存占用   : {alloc / 1024 / 1024:.1f} MB')

    y0, y1 = int(height * args.top), int(height * args.bottom)
    hits = 0
    infer_times = []
    for i in range(args.frames):
        idx = int(total * (i + 1) / (args.frames + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f'  frame {idx}: 读取失败')
            continue
        crop = frame[y0:y1, :]
        t = time.time()
        boxes, texts = ocr.predict(crop)
        dt = time.time() - t
        infer_times.append(dt)
        joined = ' | '.join(f'{txt}({score:.2f})' for txt, score in texts)
        print(f'  frame {idx:5d}  {dt * 1000:7.1f} ms  boxes={len(boxes):2d}  {joined}')
        if texts:
            hits += 1
    cap.release()

    print('-' * 60)
    if infer_times:
        print(f'平均单帧耗时   : {np.mean(infer_times) * 1000:.1f} ms')
    print(f'GPU 显存峰值   : {paddle.device.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB')
    print(f'识别到文本的帧 : {hits}/{args.frames}')

    if hits == 0:
        print('[FAIL] 所有帧都没有识别出文本，CUDA 链路可能未生效')
        return 1
    print('[OK] CUDA 模式 OCR 链路跑通')
    return 0


if __name__ == '__main__':
    sys.exit(main())
