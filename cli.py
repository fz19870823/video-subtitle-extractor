#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行入口（供 PyInstaller 打包为独立可执行文件使用）

用法：
  vse-cli.exe --video <视频路径> [--area y0,y1,x0,x1] [--output out.srt]
              [--mode accurate|fast] [--lang ch] [--no-hwaccel] [--debug-state]

退出码：0 成功 / 1 参数错误 / 2 视频不存在 / 3 提取失败
"""
import os

# 打包后 sys.frozen 为 True，此时以 exe 所在目录为工作目录，
# 保证 config/config.json、output/ 等相对路径行为与源码目录一致。
if getattr(__import__('sys'), 'frozen', False):
    import sys as _sys_frozen
    import shutil as _shutil
    _EXE_DIR = os.path.dirname(os.path.abspath(_sys_frozen.executable))
    os.chdir(_EXE_DIR)
    os.environ.setdefault('PADDLE_PDX_CACHE_HOME', os.path.join(_EXE_DIR, '.paddlex'))
    # Windows 控制台/重定向默认用 GBK（cp936）。OCR 文本里出现 © 这类字符时
    # print 会抛 UnicodeEncodeError，曾把 OCR 消费线程整个打死（producer 随后撑满
    # 队列永久阻塞，主进程 join 挂死）。统一按 UTF-8 输出，无法编码的字符转义。
    for _stream in (_sys_frozen.stdout, _sys_frozen.stderr):
        try:
            _stream.reconfigure(encoding='utf-8', errors='backslashreplace')
        except Exception:
            pass
    # CONFIG_FILE 是相对 cwd 的 'config/config.json'，目录必须存在；
    # 首次运行把打包进 _internal 的默认配置复制到 exe 同级，之后用户改的就是这份。
    try:
        os.makedirs(os.path.join(_EXE_DIR, 'config'), exist_ok=True)
        _dst_cfg = os.path.join(_EXE_DIR, 'config', 'config.json')
        _src_cfg = os.path.join(getattr(_sys_frozen, '_MEIPASS', _EXE_DIR), 'config', 'config.json')
        if not os.path.exists(_dst_cfg) and os.path.exists(_src_cfg):
            _shutil.copyfile(_src_cfg, _dst_cfg)
    except Exception:
        pass
# 跳过 paddlex 联网检查模型源，避免离线环境启动卡顿
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')


def build_parser():
    import argparse
    p = argparse.ArgumentParser(prog='vse-cli', description='视频硬字幕提取（命令行版）')
    p.add_argument('--video', required=True, help='视频文件路径')
    p.add_argument('--area', default=None,
                   help='字幕区域 y0,y1,x0,x1（像素），不填则半自动检测')
    p.add_argument('--output', default=None,
                   help='输出 srt 路径，默认与视频同目录同名')
    p.add_argument('--mode', default='accurate', choices=['accurate', 'fast'],
                   help='提取模式，默认 accurate（精确）')
    p.add_argument('--lang', default='ch', help='字幕语言，默认 ch')
    p.add_argument('--no-hwaccel', action='store_true', help='禁用硬件加速')
    p.add_argument('--debug-state', action='store_true',
                   help='写 state_events.log（含跳帧统计）')
    return p


def _patch_ocr_single_process():
    """
    冻结环境（PyInstaller）下用同进程线程替代 multiprocessing 子进程。

    原因：打包后 OCR 子进程需要重新加载整套 paddle/CUDA dll，实测会卡在启动阶段
    完全不消费任务队列，而主进程在 `subtitle_ocr_process.join()` 上无限等待
    （整集 34056 帧跑到 100% 后永久挂住，子进程 CPU 停在 4.9s）。
    线程版走完全相同的 producer/consumer 逻辑，接口与 Process 兼容，避开 mp 启动问题。
    """
    import queue as _queue
    from threading import Thread
    from types import SimpleNamespace
    from backend.tools import subtitle_ocr as _so

    class _ThreadProcess:
        """模拟 multiprocessing.Process 的最小接口（ProcessManager 会读 pid 并 join）"""

        def __init__(self, thread):
            self._t = thread
            self.pid = os.getpid()

        def join(self, timeout=None):
            self._t.join(timeout)

        def is_alive(self):
            return self._t.is_alive()

        def terminate(self):
            pass

    _required = ('REC_CHAR_TYPE', 'DROP_SCORE', 'SUB_AREA_DEVIATION_RATE',
                 'DEBUG_OCR_LOSS', 'HARDWARD_ACCELERATOR')

    def _local_async_start(video_path, raw_subtitle_path, sub_area, options):
        for key in _required:
            assert key in options, f'options缺少参数：{key}'
        task_queue = _queue.Queue()
        progress_queue = _queue.Queue()
        t = Thread(
            target=_so.subtitle_extract_handler,
            name='subtitle-ocr-local',
            args=(task_queue, progress_queue, video_path, raw_subtitle_path,
                  sub_area, SimpleNamespace(**options)),
            daemon=True,
        )
        t.start()
        return _ThreadProcess(t), task_queue, progress_queue

    _so.async_start = _local_async_start

    # 同进程里主线程与 OCR 线程各有一套 predictor，并发调用 CUDA 会直接死锁
    # （实测主进程 CPU 停住、join 永久阻塞）。给 OcrRecogniser 的构造与推理加
    # 类级可重入锁，让两者串行使用 GPU。多进程模式下不存在这个问题，故只在回退时加。
    import threading
    from backend.tools.ocr import OcrRecogniser as _OR
    if not getattr(_OR, '_vse_locked', False):
        _gpu_lock = threading.RLock()
        for _name in ('__init__', 'predict', 'recognise_polys', 'init_model'):
            _orig = getattr(_OR, _name)

            def _make(orig):
                def _wrapped(self, *a, **k):
                    with _gpu_lock:
                        return orig(self, *a, **k)
                _wrapped.__name__ = getattr(orig, '__name__', 'wrapped')
                return _wrapped

            setattr(_OR, _name, _make(_orig))
        _OR._vse_locked = True


def main(argv=None):
    import sys
    import time

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.debug_state:
        os.environ['VSE_DEBUG_STATE'] = '1'

    # 排查卡死：VSE_FAULTHANDLER=1 时每 120s 把所有线程栈写入 faulthandler.log
    if os.environ.get('VSE_FAULTHANDLER') == '1':
        import faulthandler
        _fh = open(os.path.join(os.getcwd(), 'faulthandler.log'), 'a', encoding='utf-8')
        faulthandler.dump_traceback_later(120, repeat=True, file=_fh)

    if not os.path.exists(args.video):
        print(f'[ERROR] 视频不存在: {args.video}', file=sys.stderr)
        return 2

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    from backend.config import config
    from backend.main import SubtitleExtractor
    from backend.bean.subtitle_area import SubtitleArea

    # 冻结环境默认用同进程线程跑 OCR（mp 子进程在打包后起不来）；
    # 源码环境可用 VSE_SINGLE_PROCESS=1 复现同一路径，便于对照验证。
    if getattr(sys, 'frozen', False) or os.environ.get('VSE_SINGLE_PROCESS') == '1':
        _patch_ocr_single_process()

    config.set(config.mode, args.mode)
    config.set(config.debugNoDeleteCache, False)
    config.set(config.wordSegmentation, False)
    try:
        config.set(config.language, args.lang)
    except Exception:
        pass
    if args.no_hwaccel:
        try:
            config.set(config.hardwareAcceleration, False)
        except Exception:
            pass

    area = None
    if args.area:
        try:
            y0, y1, x0, x1 = (int(v) for v in args.area.replace(' ', '').split(','))
        except ValueError:
            print('[ERROR] --area 格式应为 y0,y1,x0,x1', file=sys.stderr)
            return 1
        area = SubtitleArea(y0, y1, x0, x1)

    print(f'[INFO] video={args.video}')
    print(f'[INFO] mode={args.mode} lang={args.lang} area={args.area or "auto"} '
          f'hwaccel={not args.no_hwaccel}')

    se = SubtitleExtractor(args.video)
    se.sub_area = area

    t0 = time.time()
    try:
        se.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[ERROR] 提取失败: {e}', file=sys.stderr)
        return 3
    wall = time.time() - t0

    srt = se.subtitle_output_path
    if args.output and srt and os.path.exists(srt):
        import shutil
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if os.path.abspath(args.output) != os.path.abspath(srt):
            shutil.copyfile(srt, args.output)
        srt = args.output

    n = 0
    if srt and os.path.exists(srt):
        try:
            import pysrt
            n = len(pysrt.open(srt, encoding='utf-8'))
        except Exception:
            n = -1

    print(f'\nWALL_TIME {wall:.1f}s  entries={n}  srt={srt} '
          f'exists={bool(srt and os.path.exists(srt))}')
    import sys as _sys
    _sys.stdout.flush()
    return 0


if __name__ == '__main__':
    import multiprocessing
    # PyInstaller 下 multiprocessing(spawn) 必须调用，否则 OCR 子进程会反复重启主程序
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    import sys
    rc = main()
    sys.stdout.flush()
    # 子进程（Queue 写入端）可能残留，直接退出避免卡住
    os._exit(rc)
