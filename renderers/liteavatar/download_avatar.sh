#!/bin/bash
# LiteAvatar 頭像批次下載腳本
# 從 ModelScope Gallery 下載頭像資產並解壓至 avatars/ 目錄
#
# 使用方式:
#   1. 在瀏覽器開啟 Gallery: https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery/summary
#   2. 點選喜歡的頭像，複製瀏覽器網址列中的頭像 ID
#      例如: https://.../20250408/P1wRwMpa9BBZa1d5O9qiAsCw.png → ID = 20250408/P1wRwMpa9BBZa1d5O9qiAsCw
#   3. 下載單個頭像:
#      bash download_avatar.sh <頭像ID> <目錄名稱>
#      範例: bash download_avatar.sh 20250408/P1wRwMpa9BBZa1d5O9qiAsCw woman1
#
#   4. 批量下載（編輯下方 AVATARS 陣列後執行）:
#      bash download_avatar.sh --batch
#
# 已下載頭像列表（執行後顯示）:
#   bash download_avatar.sh --list

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AVATAR_DIR="$SCRIPT_DIR/avatars"
CACHE_DIR="$SCRIPT_DIR/.cache"

# ============================================================
# 編輯此處以批量下載： ("頭像ID:目錄名稱")
# ============================================================
AVATARS=(
  "20250408/P1wRwMpa9BBZa1d5O9qiAsCw:woman1"
  "20250612/P18qnW9QV47heC5q0hNWUiqQ:woman2"
  "20250612/P1-ripbxsCtxmzwLZZt2b6KQ:woman3"
  "20250612/P1513BdxGXLNDICglin9pfFg:woman4"
  "20250612/P1-64AzfrJY037WpS69RiUMw:woman5"
  "20250408/P1-hDQRxa5xfpZK-1yDX8PrQ:woman6"
  "20250408/P1-jdIs8de-6kDfuQG1uOc0g:woman7"
  "20250408/P11EW-z1MQ7qDBxbdFkzPPng:woman8"
  "20250408/P12On91c_j8bghkq5ZUlXSsg:woman9"
  "20250408/P149YrbMzbPkZ0HaMQ3bvNVQ:woman10"
  "20250408/P17CcHqX4clKDXK3O_iFMR8A:woman11"
)

download_one() {
  local avatar_id="$1"
  local name="$2"
  local target="$AVATAR_DIR/$name"

  if [ -f "$target/neutral_pose.npy" ]; then
    echo "  ✓ $name 已存在，跳過"
    return 0
  fi

  echo "  ⏳ 下載 $avatar_id → $name ..."
  python3 -c "
from modelscope import snapshot_download
import os, zipfile, shutil, glob

avatar_id = '$avatar_id'
name = '$name'
base = '$AVATAR_DIR'
cache = '$CACHE_DIR'

model_dir = snapshot_download(
    'HumanAIGC-Engineering/LiteAvatarGallery',
    cache_dir=cache,
    allow_patterns=[f'{avatar_id}.*'],
)
zip_path = os.path.join(model_dir, f'{avatar_id}.zip')
d = os.path.join(base, name)
os.makedirs(d, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zf:
    subdir = os.path.commonpath(zf.namelist())
    zf.extractall(d)
inner = os.path.join(d, subdir)
if os.path.isdir(inner):
    for f in os.listdir(inner):
        shutil.move(os.path.join(inner, f), os.path.join(d, f))
    os.rmdir(inner)

# Copy preview image if available
png_src = os.path.join(model_dir, f'{avatar_id}.png')
if os.path.exists(png_src):
    shutil.copy2(png_src, os.path.join(d, 'preview.png'))
    print(f'    preview.png 已複製')
print(f'    完成 ({len(os.listdir(d))} files)')
" 2>&1 | grep -v "^\(2026-\|/Users\|Processing\|Downloading\)" | head -3
}

list_avatars() {
  echo "=== 已下載頭像 ==="
  if [ ! -d "$AVATAR_DIR" ]; then
    echo "  (無)"
    return
  fi
  for d in "$AVATAR_DIR"/*/; do
    name="$(basename "$d")"
    if [ "$name" = ".cache" ]; then continue; fi
    size="$(du -sh "$d" 2>/dev/null | cut -f1)"
    files="$(ls "$d" | wc -l | tr -d ' ')"
    echo "  $name  ($size, ${files}檔)"
  done
}

case "${1:-}" in
  --batch)
    if [ ${#AVATARS[@]} -eq 0 ]; then
      echo "請先在腳本中編輯 AVATARS 陣列"
      exit 1
    fi
    for entry in "${AVATARS[@]}"; do
      id="${entry%%:*}"
      name="${entry##*:}"
      download_one "$id" "$name"
    done
    echo ""
    list_avatars
    ;;
  --list)
    list_avatars
    ;;
  "")
    if [ $# -eq 0 ]; then
      echo "用法:"
      echo "  bash download_avatar.sh <頭像ID> <目錄名稱>     # 下載單個頭像"
      echo "  bash download_avatar.sh --batch                 # 批量下載（編輯腳本）"
      echo "  bash download_avatar.sh --list                  # 列出已下載"
      echo ""
      echo "範例:"
      echo "  bash download_avatar.sh 20250408/P1wRwMpa9BBZa1d5O9qiAsCw woman1"
      exit 1
    fi
    download_one "$1" "$2"
    ;;
  *)
    download_one "$1" "$2"
    ;;
esac
