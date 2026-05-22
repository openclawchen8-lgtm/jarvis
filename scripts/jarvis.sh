#!/bin/bash
# JARVIS 管理腳本
# 用法: bash jarvis.sh start|stop|status

#Kokoro TTS 預設用 MPS（GPU）：讓 Kokoro 用 CPU 而非 MPS
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

cd "$(dirname "$0")/.."
PYTHON="/opt/homebrew/bin/python3.11"

start() {
    echo "🤖 JARVIS 啟動中..."
    screen -dmS jarvis bash -c "$PYTHON app.py 2>&1 | tee /tmp/jarvis-server.log"
    sleep 4
    screen -dmS jarvis-tg bash -c "$PYTHON bot_telegram.py 2>&1 | tee /tmp/jarvis-telegram.log"
    sleep 2
    echo "✅ Web: http://localhost:8000"
    echo "✅ Telegram: /start"
    echo "日誌: tail -f /tmp/jarvis-server.log"
}

stop() {
    echo "🛑 關閉 JARVIS..."
    # 殺掉所有 jarvis screen session
    screen -ls 2>/dev/null | grep -E "jarvis[^a-z]" | awk -F. '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null || true
    # 殺掉所有 Python app.py process
    ps aux | grep "python3.11 app.py" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    ps aux | grep -E "bot_telegram|jitsi_bridge" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    kill $(lsof -ti:8000) 2>/dev/null || true
    sleep 1
    echo "✅ 已關閉"
}

status() {
    echo "🤖 JARVIS 狀態"
    echo "================="
    lsof -i :8000 -sTCP:LISTEN 2>/dev/null | grep -q Python && echo "   ✅ Web" || echo "   ❌ Web"
    ps aux | grep bot_telegram | grep -qv grep && echo "   ✅ Telegram" || echo "   ❌ Telegram"
    ps aux | grep jitsi_bridge | grep -qv grep && echo "   🟡 Jitsi Bridge" || echo "   ⚪ Jitsi Bridge"
    echo ""
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   🩺 OK"
        curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null
    else
        echo "   🩺 無法連線"
    fi
}

restart() {
stop
sleep 2
start
sleep 15
status
tail -f /tmp/jarvis-server.log
}

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    status)  status ;;
    restart) restart ;;
    help)
         echo "用法: bash jarvis.sh <指令>"
         echo ""
         echo "指令:"
         echo "  start   啟動 Web 伺服器 + Telegram Bot"
         echo "  stop    關閉所有服務"
         echo "  status  檢查服務狀態"
         echo "  restart 重啟所有服務"
         echo "  help    顯示此說明"
         ;;
    *)       echo "用法: bash jarvis.sh <指令> （或 jarvis.sh help）" ;;

esac
