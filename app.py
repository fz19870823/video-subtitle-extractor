#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一入口：一个可执行文件同时提供图形界面和命令行。

    vse.exe                          双击 / 无参数     → 打开图形界面
    vse.exe D:\\video.mp4            只给视频路径      → 等同于 --video
    vse.exe --video a.mp4 --area ... 带参数            → 命令行提取

打包时保持 console=True：命令行模式必须能正常输出日志、进度和退出码。
图形界面模式启动的第一时间把随 exe 创建的控制台窗口隐藏掉，双击时用户
只会短暂看到窗口闪一下，不会留一个黑框在背后。
"""
import os
import sys

from backend.tools.frozen_runtime import prepare_frozen_environment

# 冻结环境：cwd 切到 exe 目录、stdio 转 UTF-8、复制默认 config
prepare_frozen_environment()
# 跳过 paddlex 联网检查模型源，避免离线环境启动卡顿
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')


def hide_console():
    """隐藏当前进程的控制台窗口（只在 Windows 的 GUI 模式下调用）。

    只在本进程独占控制台时隐藏：如果是用户从 cmd/PowerShell 里敲命令启动的，
    隐藏控制台会连带把用户的终端窗口藏起来，那种情况必须保持原样。
    """
    if os.name != 'nt':
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        hwnd = kernel32.GetConsoleWindow()
        if not hwnd:
            return
        attached = (ctypes.c_uint * 16)()
        count = kernel32.GetConsoleProcessList(attached, 16)
        if count > 1:  # 还有 cmd.exe 等进程共用该控制台
            return
        ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass


def normalize_cli_args(argv):
    """把裸视频路径归一成 --video <路径>，方便直接拖文件到 exe 上。"""
    args = list(argv[1:])
    if '--video' in args or not args:
        return args
    for i, a in enumerate(args):
        if not a.startswith('-'):
            return args[:i] + ['--video', a] + args[i + 1:]
    return args


def run_gui():
    """启动图形界面，返回 Qt 事件循环的退出码。"""
    hide_console()

    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    from PySide6 import QtCore, QtWidgets, QtGui
    from PySide6.QtCore import Qt

    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

    from gui import SubtitleExtractorGUI
    window = SubtitleExtractorGUI()
    # 先设置透明再显示，否则会有闪烁；随后用动画淡入
    window.setWindowOpacity(0.0)
    window.show()
    window.load_window_position()
    animation = QtCore.QPropertyAnimation(window, b"windowOpacity")
    animation.setDuration(300)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.start()
    return app.exec()


def main():
    if len(sys.argv) > 1:
        import cli
        rc = cli.main(normalize_cli_args(sys.argv))
        try:
            sys.stdout.flush()
        except Exception:
            pass
        # CLI 模式下可能残留 Queue 写入端线程，直接退出避免卡住
        os._exit(rc)
    return run_gui()


if __name__ == '__main__':
    import multiprocessing
    # 必须在所有逻辑之前：spawn 子进程带 --multiprocessing-fork 重新执行本 exe 时，
    # 这个调用负责接管并运行子进程任务，然后 sys.exit。
    multiprocessing.freeze_support()
    sys.exit(main())
