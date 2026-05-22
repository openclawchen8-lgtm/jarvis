"""
網路搜尋工具（使用 DuckDuckGo，免 API key）
"""

from __future__ import annotations

import urllib.request
import urllib.parse
import json
import re
import html


def search(query: str, max_results: int = 5) -> str:
    """搜尋網路並回傳摘要。"""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html_content = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"搜尋失敗：{e}"

    # 解析結果標題+連結
    results = []
    for match in re.finditer(
        r'<a rel="nofollow" class="result__a" href="(.*?)".*?>(.*?)</a>',
        html_content,
    ):
        url = match.group(1)
        title = html.unescape(re.sub(r"<.*?>", "", match.group(2)))
        results.append(f"{title}\n{url}")
        if len(results) >= max_results:
            break

    if not results:
        return "找不到結果"

    return "\n\n".join(results)
