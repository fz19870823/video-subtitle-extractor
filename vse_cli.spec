# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置：独立命令行版（onedir）

构建：
    .venv\\Scripts\\pyinstaller.exe --noconfirm vse_cli.spec
产物：
    release/vse-cli/vse-cli.exe  +  _internal/（全部依赖，可直接分发）
"""
import os
import glob
import sysconfig
from PyInstaller.utils.hooks import collect_all

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
# 必须放进 binaries 而不是 datas：onefile/onedir 下 bootloader 会把 _internal
# 加入 DLL 搜索路径，paddle 的 LoadLibrary("cudnn64_9.dll") 才能命中。
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
]

# ---- 第三方大包：整体收集（paddle 的 dll / paddlex 的动态 import 都靠这个）----
for pkg in ('paddle', 'paddlex', 'paddleocr', 'qfluentwidgets', 'wordsegment',
            'imageio_ffmpeg', 'pypdfium2'):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f'[spec] collect_all({pkg}) 跳过: {e}')

hiddenimports += [
    'backend', 'backend.main', 'backend.config', 'backend.tools.subtitle_ocr',
    'backend.tools.ocr', 'backend.tools.subtitle_detect', 'backend.bean.subtitle_area',
    'Levenshtein', 'pysrt', 'cv2', 'skimage', 'lmdb', 'pyclipper', 'shapely',
    'PIL', 'numpy', 'tqdm', 'six', 'setuptools',
]

a = Analysis(
    [os.path.join(ROOT, 'cli.py')],
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
    name='vse-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    name='vse-cli',
)
