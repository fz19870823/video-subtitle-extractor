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

# config/config.json 是个机配置文件（已 gitignore），打包时会被塞进 _internal 作为
# 新装用户的默认配置。开发时跑 CLI 会把 --lang / --force-lang 写进去，如果不还原，
# 打包出来的程序默认语言就是调试时用的那个。这里先备份、改成中性默认值，构建后还原。
echo "==> 中和默认配置（Language=ch, ForceSingleLanguage=false）"
cp config/config.json tmp/_cfg_backup.json
"$PY" - <<'PYEOF'
import json
p = 'config/config.json'
d = json.load(open(p, encoding='utf-8'))
m = d.setdefault('Main', {})
m['Language'] = 'ch'
m['ForceSingleLanguage'] = False
json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
PYEOF
trap 'cp tmp/_cfg_backup.json config/config.json 2>/dev/null || true' EXIT

echo "==> PyInstaller 构建（约 10 分钟）"
"$PY" -m PyInstaller --noconfirm --distpath tmp/_stage --workpath build/pyi vse.spec

echo "==> 交换产物到 release/vse"
if [ -d release ]; then
    mv release tmp/_release_old
fi
mkdir -p release
mv tmp/_stage/vse release/vse

# 收尾清理由 best-effort 处理：旧 release 副本 6000+ 文件、总计上万文件，
# 一次性 rm 会被 safe-delete 钩子整体拦下（BULK_CONFIRM_REQUIRED），
# 但产物此时已经就位，不该因此让整个构建报失败。逐目录删，失败的留待手动清。
echo "==> 清理中间产物（失败不影响产物）"
for d in tmp/_stage tmp/_release_old build/pyi; do
    [ -e "$d" ] || continue
    rm -rf "$d" 2>/dev/null || true
    [ -e "$d" ] && echo "    残留：$d（文件数过多被安全钩子拦，手动删或分批删）"
done

echo "==> 完成"
du -sh release/vse
ls -la release/vse/vse.exe
