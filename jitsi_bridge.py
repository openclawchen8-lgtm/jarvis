"""
JARVIS Jitsi Bridge — 加入會議室 + 文字聊天回覆 + 語音收發 (T021/T022)

啟動（文字模式）：
  /opt/homebrew/bin/python3.11 jitsi_bridge.py <room-name> [password]

啟動（語音收發模式）：
  /opt/homebrew/bin/python3.11 jitsi_bridge.py --audio <room-name> [password]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s %(name)s] %(message)s")
logger = logging.getLogger("jarvis.jitsi")

JITSI_SERVER = "https://meet.jit.si"
sys.path.insert(0, str(Path(__file__).parent))

_pipeline = None


@dataclass
class BridgeConfig:
    """Jitsi Bridge 設定。"""
    room: str = ""
    password: str = ""
    audio_enabled: bool = False       # T021/T022 啟用
    capture_enabled: bool = True      # T021: 擷取遠端音訊
    playback_enabled: bool = True     # T022: 播放 TTS 到會議
    auto_reply_chat: bool = True      # 自動回覆文字聊天
    brain_max_tokens: int = 256


async def _get_pipeline():
    """取得全域 Pipeline 實例（lazy init）。"""
    global _pipeline
    if _pipeline is None:
        from pipeline.jarvis_pipeline import JarvisPipeline, PipelineConfig
        _pipeline = JarvisPipeline(PipelineConfig(brain_max_tokens=256))
        await _pipeline.initialize()
    return _pipeline


def _send_chat(page, text: str):
    """Send text to Jitsi chat input."""
    for inp_sel, btn_sel in [
        ('[data-testid="chat-input"]', 'button[data-testid="send-message-button"]'),
        ('#chat-input', '.send-button'),
        ('input[type="text"]', 'button:has-text("Send")'),
        ('textarea', 'button:has-text("Send")'),
    ]:
        try:
            inp = page.query_selector(inp_sel)
            btn = page.query_selector(btn_sel)
            if inp and inp.is_visible() and btn and btn.is_visible():
                inp.fill(text)
                btn.click()
                return True
        except Exception:
            pass
    return False


def join_room(bridge_cfg: BridgeConfig):
    room_name = bridge_cfg.room
    password = bridge_cfg.password
    from playwright.sync_api import sync_playwright

    url = (
        f"{JITSI_SERVER}/{room_name}"
        f"#config.prejoinPageEnabled=false"
        f"&config.skipPrejoinPage=true"
        f"&config.lobbyModeEnabled=false"
        f"&config.startWithAudioMuted=true"
        f"&config.startWithVideoMuted=true"
        f"&config.channelLastN=1"
        f"&config.disableSimulcast=true"
        f"&config.p2p.enabled=false"
    )
    if password:
        url += f"&config.password={password}"
    logger.info(f"加入: {url}")

    chrome_profile = os.path.expanduser("~/Library/Application Support/Google/Chrome")
    logger.info(f"使用 Chrome profile: {chrome_profile}")

    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                user_data_dir=chrome_profile,
                headless=False,
                args=["--no-sandbox", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
                viewport={"width": 1280, "height": 720},
            )
            logger.info(f"使用 Chrome profile 成功")
        except Exception as e:
            logger.warning(f"無法使用 Chrome profile: {e}，使用一般模式")
            browser = p.chromium.launch(
                headless=False,
                args=["--no-sandbox", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
            )
            context = browser

        page = context.pages[0] if context.pages else context.new_page()
        if not context.pages:
            page = context.new_page(viewport={"width": 1280, "height": 720})
        page.goto(url)
        page.wait_for_timeout(5000)

        # ===== 處理 prejoin UI =====
        try:
            name_selectors = [
                'input[placeholder*="會議" i]', 'input[placeholder*="meeting" i]',
                'input[placeholder*="room" i]', 'input[placeholder*="名稱" i]',
                'input[name*="room"]', 'input[data-testid*="room"]',
                '#room-input', 'input[type="text"]',
            ]
            for sel in name_selectors:
                inp = page.query_selector(sel)
                if inp and inp.is_visible():
                    inp.fill(room_name)
                    logger.info(f"已填入會議名稱: {sel}")
                    break

            risk_selectors = [
                'input[type="checkbox"]', '[data-testid*="checkbox"]',
                'input[aria-checked]', '[class*="risk"] input',
                'label:has-text("風險") input',
            ]
            for sel in risk_selectors:
                try:
                    cb = page.query_selector(sel)
                    if cb and cb.is_visible() and not cb.is_checked():
                        cb.check(); break
                except Exception:
                    pass

            page.wait_for_timeout(500)
            join_selectors = [
                'button[data-testid="prejoin.joinMeeting"]',
                'button[data-testid="prejoin.joinMeetingByGo"]',
                'button:has-text("Join meeting")',
                'button:has-text("Join")',
                'button:has-text("進入")',
                'button:has-text("加入")',
                'button[data-testid*="join"]',
                'a:has-text("Join")',
                '[role="button"]:has-text("Join")',
            ]
            for sel in join_selectors:
                try:
                    btn = page.query_selector(sel)
                    if btn and btn.is_visible():
                        btn.click(); break
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"prejoin 處理: {e}")

        page.wait_for_timeout(5000)
        logger.info(f"✅ JARVIS 已加入會議室: {room_name}")

        # Inject auto-admit lobby button clicker
        page.evaluate("""
        setInterval(() => {
            document.querySelectorAll('button').forEach(b => {
                const t = (b.textContent || '').toLowerCase();
                const testid = b.getAttribute('data-testid') || '';
                const aria = (b.getAttribute('aria-label') || '').toLowerCase();
                if ((t.includes('admit') || t.includes('allow') || t.includes('join') ||
                     t.includes('允許') || t.includes('核准') || t.includes('加入') ||
                     aria.includes('admit') || aria.includes('allow') ||
                     testid.includes('lobby') || testid.includes('positive'))
                    && b.offsetParent !== null && !b.disabled) {
                    b.click();
                }
            });
        }, 2000);
        """)

        # ===== T022: Audio Playback 初始化 =====
        audio_bridge = None
        if bridge_cfg.audio_enabled:
            from jitsi_audio import JitsiAudioBridge
            audio_bridge = JitsiAudioBridge(page)

        # ===== T021: Audio Capture 初始化 =====
        capture_loop_task = None
        if bridge_cfg.audio_enabled and bridge_cfg.capture_enabled:
            # Start capture in a background thread-safe way
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                pipeline = loop.run_until_complete(_get_pipeline())
                audio_bridge.set_pipeline(pipeline)
                loop.run_until_complete(audio_bridge.start_capture())
                logger.info("T021 音訊擷取啟動 ✅")
                capture_loop_task = loop
            except Exception as e:
                logger.error(f"T021 音訊擷取啟動失敗: {e}")

        # ===== Main Loop =====
        last_reply = ""
        last_check = 0

        try:
            while True:
                # Lobby admit buttons (Python side)
                admit_selectors = [
                    'button[data-testid="lobby-admit-button"]',
                    'button[data-testid="lobby-admit-label"]',
                    'button:has-text("Admit")',
                    'button:has-text("允許")',
                    'button:has-text("核准")',
                    'button:has-text("Allow")',
                    '[aria-label*="admit" i]', '[aria-label*="allow" i]',
                    '[data-testid*="lobby"]', '[data-testid*="positive"]',
                ]
                for sel in admit_selectors:
                    try:
                        for btn in page.query_selector_all(sel):
                            if btn.is_visible() and btn.is_enabled():
                                btn.click(); break
                    except Exception:
                        pass

                # T021: Check for pending ASR responses
                if audio_bridge and audio_bridge.has_pending():
                    for resp in audio_bridge.get_pending_responses():
                        text = resp.get("response", "")
                        logger.info(f"[T021 ASR] 回覆: {text[:60]}")
                        if bridge_cfg.playback_enabled:
                            audio_bridge.say(text)
                        if bridge_cfg.auto_reply_chat:
                            _send_chat(page, text[:200])

                # Chat reply
                if bridge_cfg.auto_reply_chat:
                    try:
                        msgs = page.evaluate("""() => {
                            const sel = '[data-testid="chat-message"], .chat-message, .message-bubble, [class*="message"]';
                            const items = document.querySelectorAll(sel);
                            if (!items.length) return null;
                            return items[items.length - 1].textContent?.trim() || null;
                        }""")
                        if msgs and msgs != last_reply and msgs != "Jarvis":
                            last_reply = msgs
                            logger.info(f"收到 chat: {msgs[:80]}")
                            # Run through pipeline
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                pipeline = loop.run_until_complete(_get_pipeline())
                                result = loop.run_until_complete(pipeline.run_text(msgs))
                                reply = result.response
                            except Exception as e:
                                logger.error(f"Pipeline error: {e}")
                                reply = f"錯誤：{e}"

                            safe = reply.replace("'", "\\'").replace("\n", " ")[:200]
                            logger.info(f"回覆: {safe[:60]}")

                            _send_chat(page, safe)

                            # T022: Play TTS audio into meeting
                            if audio_bridge and bridge_cfg.playback_enabled and result and result.audio:
                                logger.info("T022: 播放 TTS 到會議")
                                audio_bridge.say_wav(result.audio)
                    except Exception:
                        pass

                # Status report
                if time.time() - last_check > 30:
                    caps = []
                    if audio_bridge: caps.append("音訊")
                    if bridge_cfg.auto_reply_chat: caps.append("聊天")
                    logger.info(f"仍監聽中 [{', '.join(caps)}] ... {room_name}")
                    last_check = time.time()

                time.sleep(2)

        except KeyboardInterrupt:
            pass
        finally:
            if audio_bridge:
                audio_bridge.stop_capture()
            try:
                context.close()
            except Exception:
                pass
            logger.info("已離開會議室")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JARVIS Jitsi Bridge")
    parser.add_argument("room", help="會議室名稱")
    parser.add_argument("password", nargs="?", default="", help="會議室密碼")
    parser.add_argument("--audio", action="store_true", help="啟用語音收發 (T021/T022)")
    parser.add_argument("--no-chat", action="store_true", help="停用文字聊天回覆")
    parser.add_argument("--no-capture", action="store_true", help="停用遠端音訊擷取")
    parser.add_argument("--no-playback", action="store_true", help="停用 TTS 播放")
    args = parser.parse_args()

    cfg = BridgeConfig(
        room=args.room,
        password=args.password or "",
        audio_enabled=args.audio,
        capture_enabled=args.audio and not args.no_capture,
        playback_enabled=args.audio and not args.no_playback,
        auto_reply_chat=not args.no_chat,
    )
    join_room(cfg)