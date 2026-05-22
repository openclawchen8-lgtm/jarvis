"""
JARVIS Telegram Bot

啟動方式：
  /opt/homebrew/bin/python3.11 bot_telegram.py

需在 ~/.jarvis_config.json 中設定：
  telegram.bot_token
  telegram.enabled = true

指令：
  /start    — 啟動訊息
  /call     — 取得 Jitsi Meet 語音通話連結
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jarvis.telegram")

# ============================================================================
# 本機 LAN IP 偵測
# ============================================================================


def get_lan_ip() -> str:
    """取得本機區網 IP，供手機/其他裝置連線使用。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

# ============================================================================
# 讀取設定
# ============================================================================

CONFIG_PATH = Path.home() / ".jarvis_config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"❌ 設定檔不存在: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


config = load_config()
tg_cfg = config.get("telegram", {})
bot_token = tg_cfg.get("bot_token", "")
allowed_users = tg_cfg.get("allowed_user_ids", [])

if not bot_token:
    print("❌ telegram.bot_token 未設定")
    sys.exit(1)

# ============================================================================
# Pipeline
# ============================================================================

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.jarvis_pipeline import JarvisPipeline, PipelineConfig

_pipeline: JarvisPipeline | None = None


async def get_pipeline() -> JarvisPipeline:
    global _pipeline
    if _pipeline is None:
        cfg = PipelineConfig(brain_model_path=os.environ.get("BRAIN_MODEL"))
        _pipeline = JarvisPipeline(cfg)
        await _pipeline.initialize()
    return _pipeline


# ============================================================================
# Telegram Bot
# ============================================================================

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters


async def start(update: Update, _context):
    await update.message.reply_text(
        "🤖 JARVIS 已就緒\n\n"
        "指令：\n"
        "/call — 取得語音通話連結\n"
        "直接傳文字或語音給我即可對話。"
    )


async def cmd_call(update: Update, _context):
    """產生一對一的 Jitsi Meet 通話連結。"""
    user_id = update.effective_user.id
    if allowed_users and user_id not in allowed_users:
        await update.message.reply_text("⛔ 你沒有使用權限")
        return

    room = f"JARVIS-{uuid.uuid4().hex[:8].upper()}"
    password = uuid.uuid4().hex[:6].upper()  # 6 位密碼
    jitsi_url = (
        f"https://meet.jit.si/{room}"
        f"#config.prejoinPageEnabled=false"
        f"&config.prejoinConfig.enabled=false"
        f"&config.skipPrejoinPage=true"
        f"&config.lobbyModeEnabled=false"
        f"&config.startWithAudioMuted=false"
        f"&config.startWithVideoMuted=true"
        f"&config.password={password}"
        f"&userInfo.displayName=%22Caller%22"
    )
    lan_ip = get_lan_ip()
    our_url = f"https://{lan_ip}:8443/call?room={room}"

    keyboard = [
        [InlineKeyboardButton("🎥 加入 Jitsi Meet 通話", url=jitsi_url)],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"📞 通話會議室已建立\n\n"
        f"會議室：{room}\n"
        f"密碼：{password}\n\n"
        f"點擊下方按鈕加入會議室，JARVIS 會在 30 秒後抵達。\n"
        f"（首次加入需輸入密碼）",
        reply_markup=reply_markup,
    )
    # 等 30 秒讓使用者先加入，再啟動 Bridge
    def _join_later():
        import time
        time.sleep(30)
        script = Path(__file__).parent / "jitsi_bridge.py"
        subprocess.Popen(
            ["/opt/homebrew/bin/python3.11", str(script), room, password],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Jitsi Bridge 已啟動: {room} password={password}")
    threading.Thread(target=_join_later, daemon=True).start()
    logger.info(f"通話連結已建立: {room} password={password}")


async def chat(update: Update, _context):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    await _check_and_reply(user_id, text, update)


async def voice(update: Update, _context):
    """處理語音訊息。"""
    user_id = update.effective_user.id
    if allowed_users and user_id not in allowed_users:
        return

    file = await update.message.voice.get_file()
    wav_bytes = await file.download_as_bytearray()

    await update.message.reply_chat_action("typing")
    try:
        pipeline = await get_pipeline()
        result = await pipeline.run_voice(bytes(wav_bytes))
        reply = result.response or f"錯誤：{result.error}"
        audio = result.audio
    except Exception as e:
        reply = f"處理失敗：{e}"
        audio = None

    if audio:
        import io
        await update.message.reply_voice(io.BytesIO(audio))
    else:
        await update.message.reply_text(reply)


async def _check_and_reply(user_id: int, text: str, update: Update):
    if allowed_users and user_id not in allowed_users:
        await update.message.reply_text("⛔ 你沒有使用權限")
        return
    if not text:
        return
    await update.message.reply_chat_action("typing")
    try:
        pipeline = await get_pipeline()
        result = await pipeline.run_text(text)
        reply = result.response or f"錯誤：{result.error}"
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        reply = f"處理失敗：{e}"
    await update.message.reply_text(reply)


def main():
    app = Application.builder().token(bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("call", cmd_call))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    app.add_handler(MessageHandler(filters.VOICE, voice))

    print("🤖 JARVIS Telegram Bot 已啟動...")
    app.run_polling()


if __name__ == "__main__":
    main()
