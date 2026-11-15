# -*- coding: utf-8 -*-
"""
@desc: PyInstaller 冻结环境（sys.frozen）下的运行时适配层

源码直接运行时这里的补丁全部不生效，行为与原版一致；只有在打包后的
exe 里才需要下面这些处理：

1. 工作目录 / 缓存 / 默认配置对齐到 exe 同级目录
   CONFIG_FILE 是相对 cwd 的 'config/config.json'，不改会让 qconfig 写到
   用户当前所在目录，甚至因为目录不存在而启动失败。
2. stdio 强制 UTF-8
   Windows 控制台/重定向默认 GBK(cp936)。OCR 文本里出现 © 这类字符时
   print 抛 UnicodeEncodeError，曾把 OCR 消费线程整个打死（producer 随后
   撑满队列永久阻塞，主进程 join 挂死）。
3. OCR 子进程改为同进程线程
   打包后 multiprocessing 子进程要重新加载整套 paddle/CUDA dll，实测会卡在
   启动阶段完全不消费任务队列（整集跑到 100% 后永久挂住）。线程版走完全相同的
   producer/consumer 逻辑，接口与 Process 兼容。
4. OCR predictor 加类级可重入锁
   同进程里主线程与 OCR 线程各有一套 predictor，并发调用 CUDA 会直接死锁
   （实测主进程 CPU 停住、join 永久阻塞）。多进程模式不存在这个问题。
"""
import os
import sys

FROZEN = bool(getattr(sys, 'frozen', False))


def exe_dir():
    """打包后 exe 所在目录；源码运行时退化为项目根目录。"""
    if FROZEN:
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def force_utf8_stdio():
    """把 stdout/stderr 重配置为 UTF-8，无法编码的字符转义而不是抛异常。

    windowed（console=False）模式下 stdout 可能是 None，reconfigure 不适用，
    直接跳过即可。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
        except Exception:
            pass


def _copy_default_config(root):
    """首次运行把打包进 _internal 的默认配置复制到 exe 同级，之后用户改的就是这份。"""
    try:
        os.makedirs(os.path.join(root, 'config'), exist_ok=True)
        dst = os.path.join(root, 'config', 'config.json')
        src = os.path.join(getattr(sys, '_MEIPASS', root), 'config', 'config.json')
        if not os.path.exists(dst) and os.path.exists(src):
            import shutil
            shutil.copyfile(src, dst)
    except Exception:
        pass


def _copy_runtime_assets(root):
    """把源码里按「相对 cwd」引用的界面资源复制到 exe 同级目录。

    gui.py:  QIcon("design/vse.ico")
    ui/icon/my_fluent_icon.py:  f'./ui/icon/stop_{color}.svg'
    这两处都是相对当前工作目录的路径，光放进 _internal 是找不到的。
    """
    import shutil
    meipass = getattr(sys, '_MEIPASS', root)
    if os.path.abspath(meipass) == os.path.abspath(root):
        return
    for rel in ('ui/icon', 'design'):
        src = os.path.join(meipass, rel)
        dst = os.path.join(root, rel)
        if not os.path.isdir(src):
            continue
        try:
            os.makedirs(dst, exist_ok=True)
            for name in os.listdir(src):
                s, d = os.path.join(src, name), os.path.join(dst, name)
                if os.path.isfile(s) and not os.path.exists(d):
                    shutil.copyfile(s, d)
        except Exception:
            pass


def prepare_frozen_environment():
    """入口进程启动时调用：把 cwd / 缓存 / stdio / 配置对齐到 exe 同级。"""
    if not FROZEN:
        return
    root = exe_dir()
    try:
        os.chdir(root)
    except Exception:
        pass
    os.environ.setdefault('PADDLE_PDX_CACHE_HOME', os.path.join(root, '.paddlex'))
    force_utf8_stdio()
    _copy_default_config(root)
    _copy_runtime_assets(root)


def apply_single_process_ocr():
    """用同进程线程替代 OCR 的 multiprocessing 子进程（幂等）。"""
    import queue as _queue
    from threading import Thread
    from types import SimpleNamespace
    from backend.tools import subtitle_ocr as _so

    class _ThreadProcess:
        """模拟 multiprocessing.Process 的最小接口（ProcessManager 会读 pid 并 join）"""

        # 关键标记：这不是真进程，pid 只能填自己（os.getpid()）。
        # 如果被 ProcessManager 收进去，terminate_all() 会执行
        # `taskkill /F /T /PID <自己的 pid>` 把当前进程直接杀掉——
        # 现象就是日志全部打完、字幕也生成好，但子进程退出码非 0，
        # 界面把任务判成失败。调用方必须跳过这种对象。
        is_local_thread = True

        def __init__(self, thread):
            self._t = thread
            self.pid = os.getpid()

        def join(self, timeout=None):
            self._t.join(timeout)

        def is_alive(self):
            return self._t.is_alive()

        def terminate(self):
            pass

        def kill(self):
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

    # 子进程里的 stdout 同样是 GBK，OCR 日志里出现 © 会把消费线程打死
    force_utf8_stdio()

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


def single_process_ocr_enabled():
    """冻结环境，或显式设置 VSE_SINGLE_PROCESS=1（便于源码复现同一路径）。"""
    return FROZEN or os.environ.get('VSE_SINGLE_PROCESS') == '1'
