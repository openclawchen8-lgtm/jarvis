"""
JARVIS 內建技能

- date_time: 當前日期時間
- weather: 天氣查詢（透過 wttr.in，需網路）
"""

from __future__ import annotations

import datetime
import logging
import urllib.request
import urllib.error
import json

from skills import Skill

logger = logging.getLogger("jarvis.skills.builtin")

# ============================================================================
# 日期時間技能
# ============================================================================

WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def date_time_handler(_input: str) -> str:
    """回傳當前日期時間，注入 prompt。"""
    now = datetime.datetime.now()
    date_str = f"{now.year}年{now.month}月{now.day}日 {WEEKDAYS[now.weekday()]}"
    time_str = f"{now.hour:02d}:{now.minute:02d}"
    return (
        f"【系統資訊】\n"
        f"當前日期：{date_str}\n"
        f"當前時間：{time_str}\n"
        f"使用者所在地：台灣（推測）\n"
        f"時區：UTC+8\n"
        f"---\n"
        f"請根據以上資訊回答使用者的問題。\n"
        f"使用者說：{_input}"
    )


date_time_skill = Skill(
    name="date_time",
    keywords=["禮拜", "星期", "日期", "今天", "時間", "幾號", "幾月", "幾年", "幾點", "時"],
    handler=date_time_handler,
)

# ============================================================================
# 天氣技能
# ============================================================================


def weather_handler(_input: str) -> str:
    """透過 wttr.in 查詢天氣（免 API key）。失敗時回傳 fallback。"""
    try:
        url = "https://wttr.in/Taipei?format=%C+%t+%h+%w&lang=zh"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            weather_text = resp.read().decode("utf-8").strip()
    except Exception as e:
        logger.warning(f"天氣查詢失敗: {e}")
        weather_text = "無法取得天氣資料（離線）"

    now = datetime.datetime.now()
    date_str = f"{now.year}年{now.month}月{now.day}日 {WEEKDAYS[now.weekday()]}"

    return (
        f"【系統資訊】\n"
        f"當前日期：{date_str}\n"
        f"當前時間：{now.hour:02d}:{now.minute:02d}\n"
        f"天氣狀況：{weather_text}\n"
        f"---\n"
        f"請根據以上資訊回答使用者的問題。\n"
        f"使用者說：{_input}"
    )


weather_skill = Skill(
    name="weather",
    keywords=["天氣", "溫度", "下雨", "颱風", "氣溫", "濕度"],
    handler=weather_handler,
)
