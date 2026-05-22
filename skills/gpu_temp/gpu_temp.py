"""
GPU 溫度查詢工具
"""

import subprocess


def handler(query: str = "") -> str:
    try:
        result = subprocess.run(
            ["sudo", "powermetrics", "--samplers", "gpu_power", "-i", "500", "-n", "1"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "GPU die temperature" in line:
                return line.strip()
        return "無法取得 GPU 溫度"
    except Exception as e:
        return f"查詢失敗：{e}"
