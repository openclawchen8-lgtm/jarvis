#!/bin/bash
#===============================================================================
# 下載 Qwen1.5-0.5B-Chat-MNN 模型（HuggingFace）
#
# 用法：
#   bash scripts/download_qwen1.5_model.sh
#
# 模型存放：~/Projects/JARVIS-on-mac/models_qwen1.5/
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$SCRIPT_DIR/models_qwen1.5"

echo "============================================"
echo "Qwen1.5-0.5B-Chat-MNN 模型下載"
echo "============================================"
echo "存放目錄：$MODELS_DIR"
echo ""

# 檢查 huggingface_hub
if ! python3 -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
    echo "❌ 缺少 huggingface_hub，正在安裝..."
    pip3 install huggingface_hub
fi

echo "開始下載..."
python3 -c "
from huggingface_hub import snapshot_download
import os

target = '$MODELS_DIR'
os.makedirs(target, exist_ok=True)

print('下載中，請稍候...')
snapshot_download(
    repo_id='taobao-mnn/Qwen1.5-0.5B-Chat-MNN',
    local_dir=target,
    local_dir_use_symlinks=False,
)
print('下載完成 ✅')
"

echo ""
echo "驗證檔案..."
REQUIRED="config.json tokenizer.txt"
# Qwen1.5 可能用不同命名
for f in config.json tokenizer.txt; do
    if [ -f "$MODELS_DIR/$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ⚠️  $f 未找到"
    fi
done

echo ""
ls -lh "$MODELS_DIR/" | head -10
echo ""
echo "完成！"
