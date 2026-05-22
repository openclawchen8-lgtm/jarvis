"""
FinMind 真實資料查詢工具

實際呼叫 FinMind API 回傳股票/匯率等資料。
需先註冊 https://finmindtrade.com 取得 token。
"""

from __future__ import annotations

import io
import json
import logging
import base64
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("jarvis.skills.finmind")

BASE = "https://api.finmindtrade.com/api/v4/data"

def _get_token() -> str:
    """從設定檔讀取 FinMind token。"""
    import json
    p = Path(__file__).parent.parent / "run" / ".." / ".." / ".jarvis_config.json"
    cfg_path = Path.home() / ".jarvis_config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                return json.load(f).get("finmind", {}).get("token", "")
        except Exception:
            return ""
    return ""
CHART_DIR = Path(__file__).parent.parent / "assets" / "charts"


def query(dataset: str, data_id: str = "", days: int = 30) -> str:
    """查詢 FinMind API 回傳 JSON 文字摘要。"""
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    params = f"dataset={dataset}&start_date={start}&end_date={end}"
    if data_id:
        params += f"&data_id={data_id}"
    url = f"{BASE}?{params}"

    headers = {}
    token = _get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return f"查詢失敗：{e}"

    records = data.get("data", [])
    if not records:
        return "沒有資料"

    lines = [f"共 {len(records)} 筆資料，顯示前 10 筆："]
    for r in records[:10]:
        lines.append(str(r))
    return "\n".join(lines)


def _fetch_data(stock_id: str, days: int) -> list:
    """查詢股價並回傳 list of dict。"""
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    url = f"{BASE}?dataset=TaiwanStockPrice&data_id={stock_id}&start_date={start}&end_date={end}"
    token = _get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read()).get("data", [])


def plot_stock_chart(stock_id: str, days: int = 60) -> str:
    """繪製股價走勢圖，回傳圖表網址。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.font_manager as fm
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    # macOS 中文字型
    for fp in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc",
               "/System/Library/Fonts/STHeiti Medium.ttc"]:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    records = _fetch_data(stock_id, days)
    if not records:
        return f"沒有 {stock_id} 的資料"

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0a0e17")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#9ca3af")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#374151")
        ax.spines["bottom"].set_color("#374151")

    dates = df["date"]
    ax1.fill_between(dates, df["min"], df["max"], alpha=0.15, color="#60a5fa", label="高低區間")
    ax1.plot(dates, df["close"], color="#f87171", linewidth=2, marker=".", label="收盤價")
    ax1.plot(dates, df["open"], color="#60a5fa", linewidth=1, linestyle="--", alpha=0.6, label="開盤價")
    ax1.set_title(f"{stock_id} 近 {days} 日股價走勢", color="#e5e7eb", fontsize=14)
    ax1.legend(loc="upper left", facecolor="#1f2937", labelcolor="#e5e7eb")
    ax1.grid(True, alpha=0.1)

    # 成交量
    ax2.bar(dates, df["Trading_Volume"] / 1e6, color="#60a5fa", alpha=0.5, width=1)
    ax2.set_ylabel("成交量 (百萬)", color="#9ca3af")

    plt.xticks(rotation=30, color="#9ca3af")
    fig.tight_layout()

    # 存檔
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{stock_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = CHART_DIR / filename
    fig.savefig(path, dpi=120, facecolor="#0a0e17")
    plt.close(fig)

    url = f"/assets/charts/{filename}"
    logger.info(f"圖表已儲存: {path}")
    return f"📈 圖表已產生：\n\n開盤價、收盤價、最高最低區間、成交量。\n查看圖表：{url}"


def taiwan_stock_price(stock_id: str, days: int = 30) -> str:
    """查詢台股股價。自動從文字中提取股票代號。"""
    import re
    ids = re.findall(r"\d{4}", stock_id)
    sid = ids[0] if ids else stock_id.strip()
    return query("TaiwanStockPrice", sid, days)


def taiwan_stock_monthly_revenue(stock_id: str, months: int = 12) -> str:
    """查詢月營收。"""
    return query("TaiwanStockMonthRevenue", stock_id, months * 30)


def exchange_rate(currency: str = "USD", days: int = 30) -> str:
    """查詢匯率。"""
    return query("TaiwanExchangeRate", currency, days)
