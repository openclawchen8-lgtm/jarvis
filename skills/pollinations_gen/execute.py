"""
pollinations_gen — 完全免費的圖像生成技能

使用 pollinations.ai，不需要 API token。

URL 格式：https://image.pollinations.ai/prompt/{encoded_prompt}
參數：
- width, height — 圖像尺寸
- model — 模型名稱 (flux, sdxl, turbo, any)
- seed — 隨機種子（可重現）
- nologo — 隱藏 logo
"""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.parse
from pathlib import Path

import requests

logger = logging.getLogger("jarvis.skill.pollinations_gen")

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_MODEL = "flux"
BASE_URL = "https://image.pollinations.ai/prompt"


def _parse_query(query: str) -> tuple[str, str, dict]:
    """解析 query，return (prompt, model, extra_kwargs)"""
    prompt = query.strip()
    model = DEFAULT_MODEL
    extra = {}

    # 解析 | 分隔的參數
    if "|" in query:
        parts = query.split("|")
        prompt = parts[0].strip()
        for part in parts[1:]:
            part = part.strip()
            if part.lower().startswith("model:"):
                model = part.split(":", 1)[1].strip().lower()
            elif part.lower().startswith("width:"):
                try:
                    extra["width"] = int(part.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif part.lower().startswith("height:"):
                try:
                    extra["height"] = int(part.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif part.lower().startswith("seed:"):
                try:
                    extra["seed"] = int(part.split(":", 1)[1].strip())
                except ValueError:
                    pass

    return prompt, model, extra


def generate(prompt: str, model: str = DEFAULT_MODEL, **kwargs) -> bytes | None:
    """呼叫 pollinations.ai 生成圖像"""
    params = {
        "width": kwargs.get("width", DEFAULT_WIDTH),
        "height": kwargs.get("height", DEFAULT_HEIGHT),
        "model": model,
        "nologo": "true",
        "seed": kwargs.get("seed"),
    }
    # 移除 None 值
    params = {k: v for k, v in params.items() if v is not None}

    encoded_prompt = urllib.parse.quote(prompt)
    url = f"{BASE_URL}/{encoded_prompt}"

    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.Timeout:
        logger.warning("pollinations.ai 請求超時")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"pollinations.ai 請求失敗: {e}")
        return None


def execute(query_data: str) -> str:
    """主入口"""
    if not query_data or not query_data.strip():
        return "錯誤：請提供提示詞"

    prompt, model, extra = _parse_query(query_data)
    if not prompt:
        return "錯誤：提示詞為空"

    logger.info(f"pollinations 生成：model={model}, prompt={prompt[:50]}...")

    image_bytes = generate(prompt, model, **extra)
    if not image_bytes:
        return "錯誤：圖像生成失敗（pollinations.ai 無法回應）"

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))

        # 轉 PNG
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        return (
            f"✅ pollinations 圖像生成成功 ({img.width}x{img.height}, model={model})\n"
            f"[IMAGE_DATA]:{b64}"
        )
    except Exception as e:
        return f"錯誤：圖像處理失敗 - {e}"


if __name__ == "__main__":
    # 測試
    result = execute("A photorealistic full-body shot of a young woman standing in Taipei, head-to-toe view, 35mm lens")
    print(result[:200] + "..." if len(result) > 200 else result)