#!/bin/bash
#===============================================================================
# 下載 Whisper.cpp GGML 模型
#
# 用法：
#   bash scripts/download_whisper_model.sh [base|base.en|tiny|small]
#   bash scripts/download_whisper_model.sh          # 預設：base
#
# 模型存放：~/Projects/JARVIS-on-mac/whisper.cpp/models/
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WHISPER_DIR="$SCRIPT_DIR/whisper.cpp"
DEV_WHISPER="$SCRIPT_DIR/dev/whisper.cpp"

MODEL="${1:-base}"

echo "============================================"
echo "Whisper.cpp GGML 模型下載"
echo "============================================"
echo "模型：$MODEL"
echo "存放目錄：$WHISPER_DIR/models/"
echo ""

# 如果有 dev/whisper.cpp，用它的下載腳本
if [ -f "$DEV_WHISPER/models/download-ggml-model.sh" ]; then
    echo "使用 dev/whisper.cpp 的下載腳本..."
    bash "$DEV_WHISPER/models/download-ggml-model.sh" "$MODEL" "$WHISPER_DIR/models"
else
    echo "❌ 找不到 dev/whisper.cpp，請先 clone whisper.cpp"
    echo "   git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git dev/whisper.cpp"
    exit 1
fi

echo ""
echo "驗證..."
if [ -f "$WHISPER_DIR/models/ggml-$MODEL.bin" ]; then
    SIZE=$(du -sh "$WHISPER_DIR/models/ggml-$MODEL.bin" | cut -f1)
    echo "  ✅ ggml-$MODEL.bin ($SIZE)"
else
    echo "  ❌ ggml-$MODEL.bin 未找到"
    ls "$WHISPER_DIR/models/" | head -5
fi

echo ""
echo "完成！"
