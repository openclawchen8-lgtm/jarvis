#!/bin/bash
# MNN 建置腳本（Metal 加速）- Javis T001
# 用法: bash setup-mnn.sh
# 依賴環境變數: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/mnn-build.log"

# Telegram 通知函數（使用環境變數）
send_telegram() {
  local message="$1"
  local token="$TELEGRAM_BOT_TOKEN"
  local chat_id="$TELEGRAM_CHAT_ID"
  if [ -z "$token" ] || [ -z "$chat_id" ]; then
    echo "Telegram: token 或 chat_id 未設定，跳過通知"
    return
  fi
  python3 -c "
import urllib.request, urllib.parse
token = '$token'
chat_id = '$chat_id'
msg = '''$message'''
url = f'https://api.telegram.org/bot{token}/sendMessage'
data = urllib.parse.urlencode({'chat_id': chat_id, 'text': msg})
try:
    req = urllib.request.Request(url, data=data.encode())
    urllib.request.urlopen(req, timeout=10)
    print('Telegram OK')
except Exception as e:
    print(f'Telegram failed: {e}')
" 2>/dev/null || echo "Telegram send failed"
}

cd "$SCRIPT_DIR"

# --- Step 1/5: 前置檢查（cmake + ninja）---
echo "[1/5] 前置檢查：確認編譯工具..." | tee -a "$LOG"
for cmd in cmake ninja; do
  if ! command -v $cmd &>/dev/null; then
    echo "⚠️ $cmd 未找到，正在透過 Homebrew 安裝..." | tee -a "$LOG"
    brew install $cmd 2>&1 | tee -a "$LOG"
    send_telegram "⚙️ 正在安裝 $cmd..."
  else
    echo "✅ $cmd 已就緒 ($(command -v $cmd))" | tee -a "$LOG"
  fi
done
echo "=== MNN 建置開始 $(date '+%Y-%m-%d %H:%M:%S') ===" | tee "$LOG"
send_telegram "🔧 MNN 建置啟動（Javis T001）..."

# --- Step 2/5: Clone MNN ---
echo "[2/5] 克隆 MNN 倉庫..." | tee -a "$LOG"
if [ -d "MNN" ]; then
  echo "MNN 已存在，略過 clone" | tee -a "$LOG"
else
  git clone --recurse-submodules https://github.com/alibaba/MNN.git 2>&1 | tee -a "$LOG"
fi

# --- Step 3/5: CMake 配置 ---
echo "[3/5] CMake 配置（Metal ON）..." | tee -a "$LOG"
mkdir -p MNN/build
cd MNN/build
CMAKE_CMD="cmake .. -G Ninja -DMNN_METAL=ON -DMNN_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release"
echo "Running: $CMAKE_CMD" | tee -a "$LOG"
$CMAKE_CMD 2>&1 | tee -a "$LOG"
METAL_STATUS=$(grep -i "METAL" CMakeCache.txt 2>/dev/null | grep -v "^#" | head -3 || echo "Metal status unknown")
echo "Metal config: $METAL_STATUS" | tee -a "$LOG"

# --- Step 4/5: 編譯 ---
echo "[4/5] 編譯（ninja -j4）..." | tee -a "$LOG"
cd "$SCRIPT_DIR/MNN/build"
ninja -j4 2>&1 | tee -a "$LOG"
BUILD_RESULT=$?

# --- Step 5/5: 驗證 ---
echo "[5/5] 驗證建置產物..." | tee -a "$LOG"
if [ -f "libMNN.a" ]; then
  SIZE=$(du -h libMNN.a 2>/dev/null | cut -f1)
  echo "✅ libMNN.a OK ($SIZE)" | tee -a "$LOG"
  send_telegram "✅ MNN 建置成功！\n📦 libMNN.a ($SIZE)\n🔧 Metal: $METAL_STATUS"
else
  echo "❌ libMNN.a NOT FOUND" | tee -a "$LOG"
  LAST_ERROR=$(tail -20 "$LOG" | grep -i "error\|failed" | tail -3)
  send_telegram "❌ MNN 建置失敗！\n🔍 $LAST_ERROR\n📝 日誌：$LOG"
  exit 1
fi

if [ -f "MNNConvert" ]; then
  echo "✅ MNNConvert OK" | tee -a "$LOG"
  send_telegram "✅ MNNConvert 已編譯"
fi

echo "=== MNN 建置完成 $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"
send_telegram "✅ MNN 建置完成！\n📝 日誌：$LOG\n⏭️ 下一個任務：T002（pymnn 安裝）"
echo "建置完成，日誌：$LOG"