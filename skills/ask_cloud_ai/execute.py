"""
ask_cloud_ai — 將複雜問題外包給雲端 AI

支援從輸入中指定雲端模型：
  使用 deepseek-ai/deepseek-v4-flash: 什麼是量子計算？
  使用 z-ai/glm-4.7-flash-free: 解釋遞迴演算法

未指定則使用 ~/.jarvis_config.json 中 ask_cloud_ai.default_model。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import requests

logger = logging.getLogger("jarvis.skill.ask_cloud_ai")

CONFIG_PATH = Path.home() / ".jarvis_config.json"


def _build_model_map() -> dict:
    """從 models.backends 建立 model → (api_base, api_key) 對照表。"""
    m = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                root = json.load(f)
            for bk in root.get("models", {}).get("backends", {}).values():
                if bk.get("type") == "openai" and bk.get("api_key"):
                    model = bk.get("model", "")
                    if model:
                        m[model] = (bk.get("api_base", ""), bk["api_key"])
        except Exception:
            pass
    return m


def _get_config() -> dict:
    """從 ~/.jarvis_config.json 讀取設定，api_base/api_key 從 models.backends 自動匹配。"""
    cfg = {
        "api_key": "",
        "api_base": "",
        "default_model": "deepseek-ai/deepseek-v4-flash",
        "fallback_model": "",
    }

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                root = json.load(f)
            aca = root.get("ask_cloud_ai", {})
            cfg["default_model"] = aca.get("default_model", cfg["default_model"])
            cfg["fallback_model"] = aca.get("fallback_model", "")
        except Exception:
            pass

    model_map = _build_model_map()
    def lookup(model: str):
        if model in model_map:
            return model_map[model]
        for b, k in model_map.values():
            return b, k
        return ("", "")

    cfg["api_base"], cfg["api_key"] = lookup(cfg["default_model"])
    cfg["_lookup"] = lookup
    return cfg


def _call(api_key: str, api_base: str, model: str, query: str, retries: int = 3) -> str:
    """呼叫單一雲端模型，失敗自動重試，成功回傳內容，全部失敗回傳 None。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一個雲端高階 AI 智囊。現在本機 7B 模型遇到技術瓶頸，將問題外包給你，請給予最深度、精準的工程解答。"},
            {"role": "user", "content": query},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    import time
    for attempt in range(retries):
        try:
            resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"{model} 第 {attempt+1} 次失敗 ({e})，2 秒後重試...")
                time.sleep(2)
            else:
                logger.warning(f"{model} 重試 {retries} 次後仍失敗")
                return None
    return None


def execute(query_data: str) -> str:
    """動態外包工具，主模型失敗自動 fallback 到備援模型。"""
    cfg = _get_config()
    api_key = cfg["api_key"]
    if not api_key:
        return "系統錯誤：未設定任何 API key"

    # 解析輸入，看是否指定了模型
    model_match = re.match(r"使用\s*(\S+)\s*[:：]\s*(.*)", query_data, re.DOTALL)
    if model_match:
        model_name = model_match.group(1).strip()
        user_query = model_match.group(2).strip()
    else:
        model_name = cfg["default_model"]
        user_query = query_data

    # 主要模型（重試 3 次）
    result = _call(cfg["api_key"], cfg["api_base"], model_name, user_query, retries=3)
    if result:
        logger.info(f"✅ 主要模型 {model_name} 成功")
        return result

    # Fallback（重試 2 次）
    fb_model = cfg.get("fallback_model", "")
    if fb_model and fb_model != model_name:
        fb_base, fb_key = cfg["_lookup"](fb_model)
        logger.info(f"🔄 切換至備援模型 {fb_model}")
        result = _call(fb_key or cfg["api_key"], fb_base or cfg["api_base"], fb_model, user_query, retries=2)
        if result:
            return result

    return f"主要模型 {model_name} 與備援模型皆失敗"
