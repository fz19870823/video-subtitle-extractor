# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置：图形界面 + 命令行统一版（onedir）

构建：
    .venv\\Scripts\\pyinstaller.exe --noconfirm vse.spec
产物：
    release_gui/vse/vse.exe  +  _internal/（全部依赖，可直接分发）

用法：
    双击 vse.exe                        → 图形界面
    vse.exe --video a.mp4 --area ...    → 命令行提取
"""
import os
import glob
import sysconfig
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(SPECPATH)  # noqa: F821  PyInstaller 注入的 spec 所在目录

datas = []
binaries = []
hiddenimports = []

# ---- 包元数据 ----
# paddlex 用 importlib.metadata 检查依赖是否"已安装"（PDFReaderBackend 需要
# opencv-contrib-python / pypdfium2 的 dist-info），冻结环境里不复制就会误报缺失。
# 全量复制最稳，dist-info 总体只有几 MB。
_SITE = sysconfig.get_paths()['purelib']
for _di in glob.glob(os.path.join(_SITE, '*.dist-info')):
    datas.append((_di, os.path.basename(_di)))

# ---- CUDA 运行库（paddle-gpu 依赖 pip 版 nvidia 包，系统里没有 cudnn 9）----
# 必须放进 binaries 而不是 datas：onedir 下 bootloader 会把 _internal 加入 DLL
# 搜索路径，paddle 的 LoadLibrary("cudnn64_9.dll") 才能命中。
# 只收 OCR 推理实际会加载的四个包，cusolver/cusparse/cufft/curand 约 1.2GB 不收。
for _nv in ('cuda_runtime', 'cublas', 'cudnn', 'nvjitlink'):
    _bin = os.path.join(_SITE, 'nvidia', _nv, 'bin')
    for _dll in glob.glob(os.path.join(_bin, '*.dll')):
        binaries.append((_dll, '.'))


# ---- 项目自身资源（路径必须与源码里的 __file__ 推导保持一致：backend/...）----
datas += [
    (os.path.join(ROOT, 'backend', 'models'), 'backend/models'),
    (os.path.join(ROOT, 'backend', 'interface'), 'backend/interface'),
    (os.path.join(ROOT, 'backend', 'configs'), 'backend/configs'),
    (os.path.join(ROOT, 'backend', 'tools', 'NotoSansCJK-Bold.otf'), 'backend/tools'),
    (os.path.join(ROOT, 'backend', 'subfinder'), 'backend/subfinder'),
    (os.path.join(ROOT, 'config', 'config.json'), 'config'),
    # 下面两项被源码用「相对 cwd」的路径引用（ui/icon/stop_*.svg、design/vse.ico），
    # 启动时 frozen_runtime 会把它们从 _internal 复制到 exe 同级目录。
    (os.path.join(ROOT, 'ui', 'icon'), 'ui/icon'),
    (os.path.join(ROOT, 'design', 'vse.ico'), 'design'),
]

# ---- 第三方大包：整体收集（paddle 的 dll / paddlex 的动态 import 都靠这个）----
for pkg in ('paddle', 'paddlex', 'paddleocr', 'qfluentwidgets', 'qframelesswindow',
            'wordsegment', 'imageio_ffmpeg', 'pypdfium2'):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f'[spec] collect_all({pkg}) 跳过: {e}')

# ui 是没有 __init__.py 的命名空间包，PyInstaller 的静态分析容易漏，显式收全
for _pkg in ('ui', 'ui.component', 'ui.icon', 'backend', 'backend.tools', 'backend.bean'):
    try:
        hiddenimports += collect_submodules(_pkg)
    except Exception as e:
        print(f'[spec] collect_submodules({_pkg}) 跳过: {e}')

hiddenimports += [
    'gui', 'cli', 'app',
    'backend.main', 'backend.config', 'backend.tools.frozen_runtime',
    'backend.tools.subtitle_ocr', 'backend.tools.ocr', 'backend.tools.subtitle_detect',
    'backend.tools.subtitle_extractor_remote_call', 'backend.tools.process_manager',
    'backend.bean.subtitle_area',
    'ui.home_interface', 'ui.advanced_setting_interface', 'ui.timeline_sync_interface',
    'ui.setting_interface', 'ui.component.video_display_component',
    'ui.component.task_list_component', 'ui.icon.my_fluent_icon',
    'showinfm', 'darkdetect', 'requests',
    'PySide6.QtSvg', 'PySide6.QtNetwork', 'PySide6.QtXml',
    'Levenshtein', 'pysrt', 'cv2', 'skimage', 'lmdb', 'pyclipper', 'shapely',
    'PIL', 'numpy', 'tqdm', 'six', 'setuptools',
]

a = Analysis(
    [os.path.join(ROOT, 'app.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'PyQt5', 'PySide2', 'matplotlib', 'notebook',
              'IPython', 'jupyter', 'pytest', 'torch'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='vse',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # 必须保留控制台：命令行模式要输出日志/进度、返回退出码。
    # 图形界面模式在 app.py 里第一时间隐藏控制台窗口。
    console=True,
    icon=os.path.join(ROOT, 'design', 'vse.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='vse',
)
