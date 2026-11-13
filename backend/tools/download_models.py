#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 PaddleOCR PP-OCRv6 推理模型到 backend/models/V6 目录。

用法示例：
    # 下载项目默认使用的两档模型（small + medium，检测 + 识别）
    python -m backend.tools.download_models

    # 只下载 small 档
    python -m backend.tools.download_models --tier small

    # 三档全下（tiny + small + medium）
    python -m backend.tools.download_models --tier all

    # 强制重新下载 / 忽略系统代理
    python -m backend.tools.download_models --force --no-proxy

模型来源（PaddleX 官方模型仓）：
    https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# 允许直接以脚本方式运行（python backend/tools/download_models.py），
# 这里不导入 backend.config，避免依赖 PySide6 / qfluentwidgets
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_URL_TEMPLATE = (
    'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/'
    'paddle3.0.0/{model_name}_infer.tar'
)

MODEL_BASE = os.path.join(BACKEND_DIR, 'models', 'V6')
DEFAULT_TIERS = ('small', 'medium')
ALL_TIERS = ('tiny', 'small', 'medium')


def build_model_names(tiers):
    """生成需要下载的模型名列表：每档的检测 + 识别模型"""
    names = []
    for tier in tiers:
        names.append(f'PP-OCRv6_{tier}_det')
        names.append(f'PP-OCRv6_{tier}_rec')
    return names


def download_file(url, dest_path, no_proxy=False):
    """带进度条下载文件"""
    if no_proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        opener = urllib.request.build_opener()

    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 vse-model-downloader'})
    with opener.open(request, timeout=120) as response:
        total = int(response.headers.get('Content-Length') or 0)
        chunk_size = 1024 * 1024
        if tqdm is not None:
            bar = tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(dest_path), leave=False)
        else:
            bar = None
        try:
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
        finally:
            if bar is not None:
                bar.close()


def extract_tar(tar_path, target_dir):
    """解压 tar 到目标目录，返回解压出的顶层目录名"""
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        # 过滤 macOS 打包产生的 AppleDouble 文件
        members = [m for m in members if not os.path.basename(m.name).startswith('._')]
        top_level = {m.name.split('/')[0] for m in members if m.name}
        try:
            # Python 3.12+ 支持 filter，避免 3.14 的默认过滤行为变更告警
            tar.extractall(target_dir, members=members, filter='data')
        except TypeError:
            tar.extractall(target_dir, members=members)
    return top_level.pop() if len(top_level) == 1 else None


def is_model_ready(model_dir):
    return os.path.exists(os.path.join(model_dir, 'inference.pdiparams'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='下载 PP-OCRv6 推理模型')
    parser.add_argument('--tier', nargs='+', default=list(DEFAULT_TIERS),
                        choices=list(ALL_TIERS) + ['all'],
                        help='要下载的模型档位，默认 small medium')
    parser.add_argument('--dest', default=MODEL_BASE, help=f'模型保存目录，默认 {MODEL_BASE}')
    parser.add_argument('--force', action='store_true', help='已存在也重新下载')
    parser.add_argument('--no-proxy', action='store_true', help='忽略系统/环境变量代理')
    args = parser.parse_args(argv)

    tiers = ALL_TIERS if 'all' in args.tier else tuple(dict.fromkeys(args.tier))
    model_names = build_model_names(tiers)

    os.makedirs(args.dest, exist_ok=True)
    print(f'模型保存目录: {args.dest}')
    print(f'待下载模型: {", ".join(model_names)}')

    failed = []
    for model_name in model_names:
        model_dir = os.path.join(args.dest, f'{model_name}_infer')
        if is_model_ready(model_dir) and not args.force:
            print(f'[跳过] {model_name} 已存在')
            continue

        url = MODEL_URL_TEMPLATE.format(model_name=model_name)
        print(f'[下载] {model_name} <- {url}')
        tmp_tar = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp:
                tmp_tar = tmp.name
            download_file(url, tmp_tar, no_proxy=args.no_proxy)

            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            top_dir = extract_tar(tmp_tar, args.dest)
            # tar 内顶层目录名形如 PP-OCRv6_small_det_infer，与目标目录一致
            if top_dir and top_dir != os.path.basename(model_dir):
                extracted = os.path.join(args.dest, top_dir)
                if os.path.exists(extracted):
                    shutil.move(extracted, model_dir)

            if is_model_ready(model_dir):
                print(f'[完成] {model_name} -> {model_dir}')
            else:
                raise RuntimeError('解压后未找到 inference.pdiparams')
        except Exception as e:
            failed.append((model_name, str(e)))
            print(f'[失败] {model_name}: {e}')
        finally:
            if tmp_tar and os.path.exists(tmp_tar):
                os.remove(tmp_tar)

    if failed:
        print('\n以下模型下载失败，请检查网络后重试（或加 --no-proxy）：')
        for name, err in failed:
            print(f'  - {name}: {err}')
        return 1

    print('\n全部模型就绪。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
