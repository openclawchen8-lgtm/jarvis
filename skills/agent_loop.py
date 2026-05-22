"""
JARVIS Agent Loop — Function Calling 引擎

支援 OpenAI 原生 function calling + 本地 MNN 關鍵字 fallback。
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("jarvis.agent")

# ============================================================================
# 技能定義類型
# ============================================================================

ToolDef = Dict[str, Any]


def load_skills(config_path: Path) -> List[ToolDef]:
    """從 ~/.jarvis_config.json 讀取 skills 定義。"""
    if not config_path.exists():
        return []
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("skills", [])
    except Exception as e:
        logger.warning(f"讀取 skills 失敗: {e}")
        return []


def to_openai_tools(skills: List[ToolDef]) -> List[dict]:
    """轉換為 OpenAI tools 格式（僅包含有 command/handler 的技能）。"""
    tools = []
    for s in skills:
        if "name" not in s:
            continue
        if "command" not in s and "handler" not in s and "handler_module" not in s:
            continue  # prompt-only skills, not function tools
        tools.append({
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return tools


def execute_tool(name: str, args: dict, skills: List[ToolDef]) -> str:
    """執行指定技能，回傳結果文字。"""
    for s in skills:
        if s["name"] != name:
            continue
        # handler 優先（直接 import 執行，免開行程）
        if "handler_module" in s and "handler_func" in s:
            try:
                import importlib
                mod = importlib.import_module(s["handler_module"])
                fn = getattr(mod, s["handler_func"])
                return fn(**args)
            except Exception as e:
                return f"handler 錯誤: {e}"
        # shell command（fallback）
        if "command" in s:
            try:
                cmd = s["command"].format(**args)
                result = subprocess.run(
                    shlex.split(cmd),
                    capture_output=True, text=True, timeout=10,
                )
                out = result.stdout.strip() or result.stderr.strip()
                return out or "執行完成（無輸出）"
            except Exception as e:
                return f"執行失敗: {e}"
        # python handler
        if "handler" in s:
            try:
                return s["handler"](**args)
            except Exception as e:
                return f"handler 錯誤: {e}"
        return f"未知技能: {name}"
    return f"技能 {name} 未定義"


# ============================================================================
# OpenAI Function Calling 對話
# ============================================================================


async def chat_with_tools(
    content: str,
    model_key: str,
    api_key: str,
    api_base: str,
    model: str,
    skills: List[ToolDef],
    system_prompt: str = "",
) -> str:
    """多輪對話支援 function calling + 提示注入技能。"""
    import openai

    client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)
    tools = to_openai_tools(skills)

    # 比對關鍵字，注入技能提示
    system_parts = []
    if system_prompt:
        system_parts.append(system_prompt)
    system_parts += [
        "你是 JARVIS，用繁體中文回答。",
        "你控制一個 3D 數位人在網頁上。",
        "可用動作：idle（站立發呆）、walk（走動）、lie_down（躺下）、prone（趴著）、talking（說話中）。",
        "當你提及姿態變化（如：休息、躺下、趴下）時，在回應最後一行加上：",
        "[ACTION: 動作名稱]",
        "例如：「我去休息一下」→ 回應內容... [ACTION: lie_down]",
        "預設不需要加 [ACTION]，只有明確表示要變換姿態時才加。",
    ]
    for s in skills:
        kws = s.get("keywords", [])
        if kws and any(kw in content for kw in kws):
            if "prompt_file" in s:
                try:
                    p = Path(__file__).parent / s["prompt_file"]
                    if p.exists():
                        system_parts.append(f"\n## {s['name']}\n{p.read_text()}")
                        logger.info(f"🔧 注入技能提示: {s['name']}")
                except Exception as e:
                    logger.warning(f"讀取 {s['prompt_file']} 失敗: {e}")

    messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    messages.append({"role": "user", "content": content})

    for _ in range(5):  # 最多 5 輪 tool call
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            max_tokens=512,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        # 有 tool call：執行並回傳結果
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(name, args, skills)
            logger.info(f"🔧 {name}({args}) → {result[:80]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
