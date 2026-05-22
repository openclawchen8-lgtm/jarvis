"""
hf_image_gen — Hugging Face 圖像生成技能

支援三種模式：
1. FLUX.1 text-to-image（Inference API）
2. InstantID face reference（Gradio Client）
3. Outpainting image extension（Gradio Client）
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger("jarvis.skill.hf_image_gen")

CONFIG_PATH = Path.home() / ".jarvis_config.json"

HF_API_BASE = "https://api-inference.huggingface.co/models"

MODELS_FLUX = {
    "schnell": "black-forest-labs/FLUX.1-schnell",
    "dev": "black-forest-labs/FLUX.1-dev",
}


def _get_config() -> dict:
    cfg = {
        "api_token": "",
        "default_model": "schnell",
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                root = json.load(f)
            hf = root.get("hf_image_gen", {})
            cfg["api_token"] = hf.get("api_token", "")
            cfg["default_model"] = hf.get("default_model", "schnell")
        except Exception:
            pass
    return cfg


def _check_gradio_client():
    try:
        from gradio_client import Client
        return True
    except ImportError:
        logger.warning("gradio_client 未安裝，安裝中...")
        import subprocess
        subprocess.run(["pip", "install", "gradio_client", "-q"], check=True)
        try:
            from gradio_client import Client
            return True
        except Exception:
            return False


# =============================================================================
# 模式一：FLUX.1 Inference API
# =============================================================================
def _generate_flux(prompt: str, model_key: str = "schnell", timeout: int = 120) -> bytes | None:
    cfg = _get_config()
    api_token = cfg["api_token"]
    if not api_token:
        return None

    model_id = MODELS_FLUX.get(model_key, MODELS_FLUX["schnell"])
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
    }

    for attempt in range(4):
        try:
            import requests
            resp = requests.post(
                f"{HF_API_BASE}/{model_id}",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 503:
                retry_after = int(resp.headers.get("Retry-After", 10))
                logger.info(f"模型載入中，等待 {retry_after}s...")
                time.sleep(min(retry_after, 30))
                continue
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.warning(f"FLUX 嘗試 {attempt+1} 失敗: {e}")
            if attempt < 3:
                time.sleep(3)
    return None


# =============================================================================
# 模式二：InstantID（Gradio Client）
# =============================================================================
def _generate_instantid(prompt: str, face_image_path: str, identity_scale: float = 0.8) -> bytes | None:
    if not _check_gradio_client():
        return None

    from gradio_client import Client, handle_file

    cfg = _get_config()
    if not cfg["api_token"]:
        return None

    try:
        client = Client("InstantX/InstantID", hf_token=cfg["api_token"])

        result = client.predict(
            face_image=handle_file(face_image_path),
            prompt=prompt,
            negative_prompt="deformed, blurry, low quality, cartoon, 3d render, bad anatomy",
            style_name="(No style)",
            num_steps=30,
            identity_scale=identity_scale,
            adapter_strength=0.8,
            api_name="/submit",
        )

        if result and hasattr(result, '__iter__'):
            for item in result:
                if isinstance(item, str) and (item.endswith('.png') or item.endswith('.jpg')):
                    with open(item, 'rb') as f:
                        return f.read()
                elif hasattr(item, 'read'):
                    return item.read()

        return None
    except Exception as e:
        logger.warning(f"InstantID 生成失敗: {e}")
        return None


# =============================================================================
# 模式三：Outpainting（Gradio Client）
# =============================================================================
def _generate_outpaint(
    prompt: str,
    image_path: str,
    overlap_percentage: int = 15,
    direction: str = "down",
) -> bytes | None:
    if not _check_gradio_client():
        return None

    from gradio_client import Client, handle_file

    cfg = _get_config()
    if not cfg["api_token"]:
        return None

    try:
        client = Client("fffiloni/multi-dataset-outpainting", hf_token=cfg["api_token"])

        result = client.predict(
            image=handle_file(image_path),
            prompt=prompt,
            negative_prompt="deformed, blurry, low quality, cartoon, 3d render",
            overlap_percentage=overlap_percentage,
            direction=direction,
            api_name="/predict",
        )

        if result and hasattr(result, '__iter__'):
            for item in result:
                if isinstance(item, str) and (item.endswith('.png') or item.endswith('.jpg')):
                    with open(item, 'rb') as f:
                        return f.read()
                elif hasattr(item, 'read'):
                    return item.read()

        return None
    except Exception as e:
        logger.warning(f"Outpainting 生成失敗: {e}")
        return None


# =============================================================================
# 主函式
# =============================================================================
def execute(query_data: str) -> str:
    cfg = _get_config()

    if not cfg["api_token"]:
        return (
            "錯誤：未設定 Hugging Face API token。\n"
            "請在 ~/.jarvis_config.json 的 hf_image_gen.api_token 設定你的 HF token。\n"
            "取得方式：https://huggingface.co/settings/tokens"
        )

    query_data = query_data.strip()
    if not query_data:
        return "錯誤：請提供提示詞"

    # 解析模式
    mode = "flux"
    prompt = query_data
    face_path = None
    image_path = None
    extra_args = {}

    # InstantID 模式
    if query_data.lower().startswith("instantid"):
        mode = "instantid"
        parts = query_data.split("|")
        prompt = parts[0][len("instantid"):].strip()
        for part in parts[1:]:
            if part.strip().startswith("face:"):
                face_path = part.split("face:", 1)[1].strip()
            elif part.strip().startswith("scale:"):
                try:
                    extra_args["identity_scale"] = float(part.split("scale:", 1)[1].strip())
                except ValueError:
                    pass

    # Outpaint 模式
    elif query_data.lower().startswith("outpaint"):
        mode = "outpaint"
        parts = query_data.split("|")
        prompt = parts[0][len("outpaint"):].strip()
        for part in parts[1:]:
            if part.strip().startswith("image:"):
                image_path = part.split("image:", 1)[1].strip()
            elif part.strip().startswith("overlap:"):
                try:
                    extra_args["overlap_percentage"] = int(part.split("overlap:", 1)[1].strip())
                except ValueError:
                    pass
            elif part.strip().startswith("direction:"):
                extra_args["direction"] = part.split("direction:", 1)[1].strip()

    # FLUX 模式（預設）
    elif query_data.lower().startswith("flux"):
        prompt = query_data[len("flux"):].strip()

    # 執行生成
    image_bytes = None
    model_label = ""

    if mode == "flux":
        model_key = cfg.get("default_model", "schnell")
        logger.info(f"FLUX 生成，使用模型: {model_key}")
        logger.info(f"Prompt: {prompt[:100]}...")
        image_bytes = _generate_flux(prompt, model_key)
        model_label = f"FLUX.1-{model_key}"

    elif mode == "instantid":
        if not face_path:
            return "錯誤：InstantID 模式需要提供 face 圖片路徑，格式：`InstantID: <prompt> | face: <路徑>`"
        logger.info(f"InstantID 生成，人臉: {face_path}")
        image_bytes = _generate_instantid(
            prompt,
            face_path,
            identity_scale=extra_args.get("identity_scale", 0.8),
        )
        model_label = "InstantID"

    elif mode == "outpaint":
        if not image_path:
            return "錯誤：Outpaint 模式需要提供 image 路徑，格式：`Outpaint: <prompt> | image: <路徑>`"
        logger.info(f"Outpainting 生成，圖片: {image_path}")
        image_bytes = _generate_outpaint(
            prompt,
            image_path,
            overlap_percentage=extra_args.get("overlap_percentage", 15),
            direction=extra_args.get("direction", "down"),
        )
        model_label = "Outpainting"

    # 處理結果
    if not image_bytes:
        return f"錯誤：{model_label} 圖像生成失敗"

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))

        # 轉換並壓縮（避免過大）
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        return (
            f"✅ {model_label} 圖像生成成功 ({img.width}x{img.height})\n"
            f"[IMAGE_DATA]:{b64}"
        )
    except Exception as e:
        return f"錯誤：圖像處理失敗 - {e}"