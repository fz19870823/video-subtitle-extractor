#!/usr/bin/env bash
# 打包 GUI + CLI 单文件程序，产物固定输出到 release/vse/
#
# 为什么不在 release 里直接构建：
#   PyInstaller onedir 会先删除目标目录，release/vse 有 6000+ 文件、5GB，
#   Windows 的回收站/安全钩子挪不动这么大的目录，会直接报错中断。
#   所以先构建到 tmp/_stage，成功后再整体换进去；旧目录改名为
#   tmp/_release_old 后删除（tmp 在 .gitignore 里，不进回收站）。
#
# 用法：bash build_release.sh
set -euo pipefail

cd "$(dirname "$0")"

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/Scripts/python.exe"

echo "==> 清理暂存目录"
rm -rf tmp/_stage tmp/_release_old
mkdir -p tmp

echo "==> PyInstaller 构建（约 10 分钟）"
"$PY" -m PyInstaller --noconfirm --distpath tmp/_stage --workpath build/pyi vse.spec

echo "==> 交换产物到 release/vse"
if [ -d release ]; then
    mv release tmp/_release_old
fi
mkdir -p release
mv tmp/_stage/vse release/vse
rm -rf tmp/_stage tmp/_release_old build/pyi

echo "==> 完成"
du -sh release/vse
ls -la release/vse/vse.exe
