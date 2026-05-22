#!/bin/bash
# JARVIS 環境自動安裝腳本
# 用法: bash scripts/setup.sh
#
# 執行步驟:
#   1. 安裝 Homebrew 套件（ffmpeg, cmake 等）
#   2. 安裝 Python 相依套件
#   3. 下載 LiteAvatar 模型權重
#   4. 編譯 MNN（選用 Metal 加速）
#   5. 下載 Whisper 語音模型

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/opt/homebrew/bin/python3.11"
LOG="$PROJECT_DIR/setup.log"

echo "=== JARVIS 環境安裝 $(date) ===" | tee "$LOG"

# ─── Step 1: Homebrew 工具 ───
echo "" | tee -a "$LOG"
echo "[1/5] Homebrew 系統套件..." | tee -a "$LOG"
brew install ffmpeg cmake ninja 2>&1 | tee -a "$LOG"

# ─── Step 2: Python 相依 ───
echo "" | tee -a "$LOG"
echo "[2/5] Python 相依套件..." | tee -a "$LOG"
$PYTHON -m pip install --upgrade pip 2>&1 | tee -a "$LOG"
$PYTHON -m pip install -r "$PROJECT_DIR/requirements.txt" 2>&1 | tee -a "$LOG"

# ─── Step 3: LiteAvatar 模型 ───
echo "" | tee -a "$LOG"
echo "[3/5] LiteAvatar 模型權重..." | tee -a "$LOG"
bash "$PROJECT_DIR/renderers/liteavatar/download_models.sh" 2>&1 | tee -a "$LOG"

# ─── Step 4: MNN 編譯（選用） ───
echo "" | tee -a "$LOG"
echo "[4/5] MNN 編譯（可略過，按 Ctrl+C 跳過）..." | tee -a "$LOG"
if [ -f "$PROJECT_DIR/MNN/build/libMNN.a" ]; then
    echo "  ✅ MNN 已編譯，跳過" | tee -a "$LOG"
else
    echo "  ⏳ 開始編譯 MNN（約 10-15 分鐘）..." | tee -a "$LOG"
    bash "$SCRIPT_DIR/setup-mnn.sh" 2>&1 | tee -a "$LOG"
fi

# ─── Step 5: Whisper 模型 ───
echo "" | tee -a "$LOG"
echo "[5/5] Whisper 語音模型下載..." | tee -a "$LOG"
bash "$SCRIPT_DIR/download_whisper_model.sh" 2>&1 | tee -a "$LOG"

# ─── 完成 ───
echo "" | tee -a "$LOG"
echo "=== 安裝完成！ ===" | tee -a "$LOG"
echo "啟動方式: bash scripts/jarvis.sh start" | tee -a "$LOG"
echo "日誌: tail -f /tmp/jarvis-server.log" | tee -a "$LOG"
