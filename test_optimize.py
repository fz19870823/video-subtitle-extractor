# -*- coding: utf-8 -*-
"""
优化对比测试驱动：
  VSE_SIG_MAX_FRAC=0  -> 基线（签名跳帧关闭，仅块内 rec batch 生效）
  默认                -> 优化版（签名跳帧 + rec batch）
用法: python test_optimize.py <video> <tag>
"""
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')
    import os
    import sys
    import time
    import shutil

    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, root)
    os.chdir(root)

    from backend.config import config
    from backend.main import SubtitleExtractor
    from backend.bean.subtitle_area import SubtitleArea

    clip = sys.argv[1]
    tag = sys.argv[2]

    config.set(config.mode, 'accurate')
    config.set(config.debugNoDeleteCache, True)
    config.set(config.wordSegmentation, False)

    se = SubtitleExtractor(clip)
    if len(sys.argv) > 3:
        y0, y1, x0, x1 = (int(v) for v in sys.argv[3].split(','))
        se.sub_area = SubtitleArea(y0, y1, x0, x1)
    else:
        se.sub_area = SubtitleArea(1684, 2138, 192, 3648)
    t0 = time.time()
    se.run()
    wall = time.time() - t0

    src = se.subtitle_output_path
    dst = os.path.join(root, f'opt_test_{tag}.srt')
    if src and os.path.exists(src):
        shutil.copyfile(src, dst)
    raw = se.raw_subtitle_path
    if raw and os.path.exists(raw):
        shutil.copyfile(raw, os.path.join(root, f'opt_test_{tag}_raw.txt'))
    print(f'\nWALL_TIME {wall:.1f}s  srt={dst} exists={os.path.exists(dst)}')
    sys.stdout.flush()
    os._exit(0)
