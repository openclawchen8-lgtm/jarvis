"""
JARVIS Jitsi Bridge — 加入會議室 + 文字聊天回覆

啟動：
  /opt/homebrew/bin/python3.11 jitsi_bridge.py <room-name> [password]
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("jarvis.jitsi")

JITSI_SERVER = "https://meet.jit.si"
sys.path.insert(0, str(Path(__file__).parent))

_pipeline = None


def ask(text: str) -> str:
    global _pipeline
    if _pipeline is None:
        from pipeline.jarvis_pipeline import JarvisPipeline, PipelineConfig
        _pipeline = JarvisPipeline(PipelineConfig(brain_max_tokens=256))
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_pipeline.initialize())
    loop = asyncio.new_event_loop()
    r = loop.run_until_complete(_pipeline.run_text(text))
    return r.response or f"錯誤：{r.error}"


def join_room(room_name: str, password: str = None):
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

        # ===== 處理會議名稱輸入 + 勾選 + 加入 =====
        try:
            # 填入會議名稱
            name_selectors = [
                'input[placeholder*="會議" i]',
                'input[placeholder*="meeting" i]',
                'input[placeholder*="room" i]',
                'input[placeholder*="名稱" i]',
                'input[name*="room"]',
                'input[data-testid*="room"]',
                '#room-input',
                'input[type="text"]',
            ]
            for sel in name_selectors:
                inp = page.query_selector(sel)
                if inp and inp.is_visible():
                    inp.fill(room_name)
                    logger.info(f"已填入會議名稱: {sel}")
                    break

            # 勾選「我了解風險」checkbox
            risk_selectors = [
                'input[type="checkbox"]',
                '[data-testid*="checkbox"]',
                'input[aria-checked]',
                '[class*="risk"] input',
                'label:has-text("風險") input',
            ]
            for sel in risk_selectors:
                try:
                    cb = page.query_selector(sel)
                    if cb and cb.is_visible() and not cb.is_checked():
                        cb.check()
                        logger.info(f"已勾選風險確認: {sel}")
                        break
                except Exception:
                    pass

            page.wait_for_timeout(300)

            # 點擊進入/加入
            page.wait_for_timeout(500)
            for sel in join_selectors:
                try:
                    btn = page.query_selector(sel)
                    if btn and btn.is_visible():
                        btn.click()
                        logger.info(f"已點擊進入: {sel}")
                        break
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"prejoin 處理: {e}")

        page.wait_for_timeout(5000)
        logger.info(f"✅ JARVIS 已加入會議室: {room_name}")

        # 注入 JS：每 2 秒嘗試自動點擊核准按鈕
        page.evaluate("""
        setInterval(() => {
            document.querySelectorAll('button').forEach(b => {
                const t = (b.textContent || '').toLowerCase();
                const testid = b.getAttribute('data-testid') || '';
                const aria = (b.getAttribute('aria-label') || '').toLowerCase();
                if ((t.includes('admit') || t.includes('允許') || t.includes('核准') ||
                     t.includes('allow') || t.includes('join') || t.includes('加入') ||
                     aria.includes('admit') || aria.includes('allow') ||
                     testid.includes('lobby') || testid.includes('positive'))
                    && b.offsetParent !== null && !b.disabled) {
                    console.log('JS click:', t.slice(0, 30), testid);
                    b.click();
                }
            });
        }, 2000);
        """)

        last_reply = ""
        last_check = 0

        try:
            while True:
                # Python：檢查並點擊核准按鈕
                admit_selectors = [
                    'button[data-testid="lobby-admit-button"]',
                    'button[data-testid="lobby-admit-label"]',
                    'button:has-text("Admit")',
                    'button:has-text("允許")',
                    'button:has-text("核准")',
                    'button:has-text("Allow")',
                    '[aria-label*="admit" i]',
                    '[aria-label*="allow" i]',
                    '[data-testid*="lobby"]',
                    '[data-testid*="positive"]',
                ]
                for sel in admit_selectors:
                    try:
                        btns = page.query_selector_all(sel)
                        for btn in btns:
                            if btn.is_visible() and btn.is_enabled():
                                t = btn.text_content() or ""
                                logger.info(f"點擊核准: {sel} '{t[:20]}'")
                                btn.click()
                                break
                    except Exception:
                        pass

                # 檢查並回覆聊天
                try:
                    msgs = page.evaluate("""() => {
                        const sel = '[data-testid="chat-message"], .chat-message, .message-bubble, [class*="message"]';
                        const items = document.querySelectorAll(sel);
                        if (!items.length) return null;
                        const last = items[items.length - 1];
                        return last.textContent?.trim() || null;
                    }""")
                    if msgs and msgs != last_reply and msgs != "Jarvis":
                        last_reply = msgs
                        logger.info(f"收到: {msgs[:80]}")
                        reply = ask(msgs)
                        safe = reply.replace("'", "\\'").replace("\n", " ")[:200]
                        logger.info(f"回覆: {safe[:60]}")

                        # 嘗試發送
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
                                    inp.fill(safe)
                                    btn.click()
                                    logger.info("已發送到 Jitsi chat")
                                    break
                            except Exception:
                                pass
                except Exception:
                    pass

                # 每 30 秒報告狀態
                if time.time() - last_check > 30:
                    try:
                        state = page.evaluate("""() => {
                            const btns = document.querySelectorAll('button');
                            const info = {url: window.location.href.slice(0,60), buttons: []};
                            btns.forEach((b, i) => {
                                if (b.offsetParent === null) return;
                                info.buttons.push(`#${i} '${(b.textContent||'').trim().slice(0,20)}' ${b.getAttribute('data-testid')||''}`);
                            });
                            return info;
                        }""")
                        logger.info(f"仍監聽中... url={state['url']}")
                        logger.info(f"按鈕: {' | '.join(state['buttons'][:8])}")
                    except Exception:
                        pass
                    logger.info(f"last reply: {last_reply[:30] if last_reply else 'none'}")
                    last_check = time.time()

                time.sleep(2)

        except KeyboardInterrupt:
            pass
        finally:
            try:
                context.close()
            except Exception:
                pass
            logger.info("已離開會議室")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 jitsi_bridge.py <room-name> [password]")
        sys.exit(1)
    room = sys.argv[1]
    password = sys.argv[2] if len(sys.argv) > 2 else None
    join_room(room, password)