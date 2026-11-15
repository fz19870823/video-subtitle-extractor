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

from backend.tools.frozen_runtime import (
    prepare_frozen_environment, apply_single_process_ocr, single_process_ocr_enabled,
)

# 打包后 sys.frozen 为 True：以 exe 所在目录为工作目录，stdio 转 UTF-8，
# 并把默认 config/config.json 复制到 exe 同级（详见 frozen_runtime 模块注释）。
prepare_frozen_environment()
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
    p.add_argument('--mode', default='accurate', choices=['auto', 'accurate', 'fast'],
                   help='提取模式，默认 accurate（精准）。auto 在有 GPU 时等同 accurate，'
                        '无 GPU 时退回 fast')
    p.add_argument('--lang', default='ch', help='字幕语言，默认 ch')
    p.add_argument('--force-lang', action='store_true',
                   help='强制单一语言：识别结果不是 --lang 指定的语言就丢弃'
                        '（过滤招牌/台标/外文logo）')
    p.add_argument('--no-hwaccel', action='store_true', help='禁用硬件加速')
    p.add_argument('--batch', type=int, default=None,
                   help='OCR 批量帧数（一次送进检测+识别模型的图片数），不填则读配置 MaxBatchSize')
    p.add_argument('--debug-state', action='store_true',
                   help='写 state_events.log（含跳帧统计）')
    p.add_argument('--via-gui-pipeline', action='store_true',
                   help='用图形界面相同的多进程链路执行（诊断打包环境用）')
    return p


def _run_via_gui_pipeline(args, area):
    """按图形界面的调用链跑一次提取：multiprocessing 子进程 + 队列回调。

    界面上点「运行」走的就是这条链路（HomeInterface.run_subtitle_extractor_process）。
    打包后用它验证 spawn 子进程、队列通信、子进程内的 OCR 线程是否都正常。
    """
    import multiprocessing
    from ui.home_interface import HomeInterface
    from backend.tools.subtitle_extractor_remote_call import SubtitleExtractorRemoteCall

    caller = SubtitleExtractorRemoteCall()
    caller.register_log_callback(lambda *a: print('[child]', *a, flush=True))
    caller.register_error_callback(lambda e: print('[child][ERROR]', repr(e), flush=True))
    caller.register_update_progress_callback(lambda *a: None)
    caller.register_manage_process_callback(lambda pid: None)

    proc = multiprocessing.Process(
        target=HomeInterface.extractor_process,
        args=(caller.queue, args.video, args.output, {'sub_area': area, 'ab_sections': None}),
    )
    proc.start()
    proc.join()
    print(f'[INFO] child_exitcode={proc.exitcode}', flush=True)
    caller.stop()
    return 0 if proc.exitcode == 0 else 3


def _patch_ocr_single_process():
    """兼容旧调用：实现已移到 backend/tools/frozen_runtime.py（GUI 共用同一份）。"""
    apply_single_process_ocr()


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
    if single_process_ocr_enabled():
        apply_single_process_ocr()

    # config.set 会触发 qconfig.save 写 config/config.json。文件被占用/只读时
    # 会抛 PermissionError（实测踩过一次：config.json 被锁，CLI 直接崩在启动阶段，
    # 界面里完全看不出原因），这里降级为警告并继续用内存里的值。
    def _set(item, value):
        try:
            config.set(item, value)
        except Exception as e:
            print(f'[WARN] 写配置文件失败（继续使用内存值）: {e}', file=sys.stderr)

    _set(config.mode, args.mode)
    _set(config.debugNoDeleteCache, False)
    _set(config.wordSegmentation, False)
    _set(config.language, args.lang)
    # 与 --mode / --lang 一致：命令行是唯一来源，不传即关闭（否则配置文件里的
    # 旧值会静默影响本次运行的结果）
    _set(config.forceSingleLanguage, bool(args.force_lang))
    if args.batch:
        # 只影响本次运行，不写回配置文件
        os.environ['VSE_MAX_BATCH'] = str(max(1, int(args.batch)))
    if args.no_hwaccel:
        _set(config.hardwareAcceleration, False)

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

    if args.via_gui_pipeline:
        # 诊断用：走图形界面同一条多进程链路，能复现"控制台完成但界面判失败"这类问题
        return _run_via_gui_pipeline(args, area)

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
