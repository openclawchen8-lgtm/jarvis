"""
web_search — 網路搜尋 + 台股股價查詢

使用 Yahoo Finance 查股價，DuckDuckGo 搜一般網頁。
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.parse

logger = logging.getLogger("jarvis.skill.web_search")


def execute(query: str) -> str:
    # 台股查詢走 Yahoo Finance
    stock_match = re.search(r"(台積電|2330|TSMC)", query)
    if stock_match:
        try:
            url = (
                "https://query1.finance.yahoo.com/v8/finance/chart/2330.TW"
                "?range=5d&interval=1d"
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            meta = data["chart"]["result"][0]["meta"]
            price = meta["regularMarketPrice"]
            prev = meta["chartPreviousClose"]
            change = price - prev
            pct = (change / prev) * 100
            return (
                f"台積電 (2330.TW) 即時股價：{price:.2f} USD（台股換算）\n"
                f"前收盤：{prev:.2f} | 漲跌：{change:+.2f} ({pct:+.2f}%)\n"
                f"資料來源：Yahoo Finance"
            )
        except Exception as e:
            logger.warning(f"Yahoo Finance 失敗: {e}")

    # 一般搜尋走 DuckDuckGo
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        results = []
        for m in re.finditer(
            r'<a rel="nofollow" class="result__a" href="(.*?)".*?>(.*?)</a>',
            html,
        ):
            title = re.sub(r"<.*?>", "", m.group(2))
            results.append(f"{title}\n{m.group(1)}")
            if len(results) >= 5:
                break

        if results:
            return "\n\n".join(results)
        return "找不到結果"
    except Exception as e:
        return f"搜尋失敗: {e}"
