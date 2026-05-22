"""
JARVIS LangChain 轉接層

把 MNN-LLM 包裝成 langchain BaseChatModel + 自訂 Agent Loop。
工具由 auto_scan_and_load_skills() 動態掃描 skills/ 目錄載入。
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import Tool

logger = logging.getLogger("jarvis.langchain")

# ============================================================================
# MNN Chat Model Wrapper
# ============================================================================


class MNNChatModel(BaseChatModel):
    """把 MNN-LLM 包成 langchain BaseChatModel。"""

    model_path: str = ""
    _llm: Any = None

    def __init__(self, model_path: str = "", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path

    def bind_tools(self, tools, **kwargs):
        return self

    def _load(self):
        if self._llm is None:
            from MNN.llm import create
            cfg = self.model_path
            if cfg and not cfg.endswith(".json"):
                cfg = cfg.rstrip("/") + "/config.json"
            elif not cfg:
                import os
                cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_qwen1.5", "config.json")
            self._llm = create(cfg)
            self._llm.load()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._load()
        parts = []
        for m in messages:
            if isinstance(m, HumanMessage):
                parts.append(f"<|im_start|>user\n{m.content}<|im_end|>")
            elif isinstance(m, AIMessage):
                parts.append(f"<|im_start|>assistant\n{m.content}<|im_end|>")
            elif isinstance(m, ToolMessage):
                parts.append(f"工具回傳：{m.content}")
            elif isinstance(m, SystemMessage):
                parts.append(f"<|im_start|>system\n{m.content}<|im_end|>")
            else:
                parts.append(str(m.content))
        parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)
        self._llm.reset()
        text = self._llm.response(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text.strip()))])

    @property
    def _llm_type(self) -> str:
        return "mnn-chat"


# ============================================================================
# 1. 自動掃描與動態 import 執行腳本 (Plug & Play 核心)
# ============================================================================


def auto_scan_and_load_skills(skills_dir_path: str = "./skills") -> list:
    """
    自動掃描 skills 目錄，讀取 SKILL.md 並動態 import execute.py 內的 execute 函數
    """
    loaded_tools = []
    skills_dir = Path(skills_dir_path)

    if not skills_dir.exists():
        logger.warning(f"⚠️ 找不到 skills 目錄: {skills_dir_path}")
        return loaded_tools

    logger.info("📡 [系統掃描] 開始自Skills目錄動態加載技能...")

    # 也從 JSON config 載入（作為 fallback）
    cfg_path = Path.home() / ".jarvis_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        for s in cfg.get("skills", []):
            name = s.get("name", "")
            desc = s.get("description", "")
            hm = s.get("handler_module", "")
            hf = s.get("handler_func", "")
            if hm and hf:
                try:
                    mod = importlib.import_module(hm)
                    fn = getattr(mod, hf)
                    loaded_tools.append(Tool(name=name, func=fn, description=desc))
                    logger.info(f"✅ [JSON 載入] skill: {name}")
                except Exception as e:
                    logger.warning(f"JSON skill {name} 載入失敗: {e}")

    for skill_folder in skills_dir.iterdir():
        if skill_folder.is_dir():
            md_path = skill_folder / "SKILL.md"
            py_path = skill_folder / "execute.py"

            if md_path.exists() and py_path.exists():
                try:
                    skill_description = md_path.read_text(encoding="utf-8")
                    skill_name = skill_folder.name

                    spec = importlib.util.spec_from_file_location(f"dynamic_skill_{skill_name}", py_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if not hasattr(module, "execute"):
                        logger.error(f"❌ {py_path.name} 中未找到 'execute' 函數，跳過")
                        continue

                    target_func = getattr(module, "execute")
                    t = Tool(name=skill_name, func=target_func, description=skill_description)
                    loaded_tools.append(t)
                    logger.info(f"✅ [技能解鎖] 成功載入: [{skill_name}]")

                except Exception as e:
                    logger.error(f"❌ 動態載入技能 [{skill_folder.name}] 失敗: {e}")

    return loaded_tools


# ============================================================================
# 2. 整合時間注入與動態工具的純 Python Agent Loop
# ============================================================================


def create_jarvis_agent(mnn_llm, tools: list, system_prompt: str = ""):
    """用純 Python 迴圈硬幹的 Jarvis 代理人 (動態工具相容版)"""
    tools_map = {t.name: t for t in tools}

    def agent_executor(user_input: str) -> str:
        import datetime
        import json
        from pathlib import Path

        # 讀取 debug flag
        debug = False
        cfg_path = Path.home() / ".jarvis_config.json"
        if cfg_path.exists():
            try:
                debug = json.loads(cfg_path.read_text()).get("debug", False)
            except Exception:
                pass

        now = datetime.datetime.now()
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(SystemMessage(
            content=(
                f"【目前環境時間】: {current_time_str}。\n"
                f"請以此時間做為解讀『今天、最近、當前』等時間詞彙的基準點。\n"
                f"可用工具：{', '.join(t.name for t in tools)}"
            )
        ))
        messages.append(HumanMessage(content=user_input))

        for step in range(5):
            ai_msg = mnn_llm.invoke(messages)
            messages.append(ai_msg)

            if debug:
                logger.info(f"─── Step {step+1} ───")
                logger.info(f"🤖 Qwen2.5: {ai_msg.content[:200] if ai_msg.content else '(tool call)'}")

            tool_executed = False

            # A. 攔截 tool_calls
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                for tc in ai_msg.tool_calls:
                    t_name = tc["name"]
                    t_args = tc["args"]
                    if t_name in tools_map:
                        result = tools_map[t_name].invoke(t_args)
                        if debug:
                            print(f"🔧 Tool [{t_name}] → {str(result)[:150]}")
                        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                        tool_executed = True
                if tool_executed:
                    continue

            # B. 文字型 Action Input
            if ai_msg.content and "Action Input:" in ai_msg.content:
                try:
                    keyword = ai_msg.content.split("Action Input:")[-1].strip().strip("\"' \n")
                    if debug:
                        print(f"🔧 Action Input: {keyword}")
                    if tools:
                        result = tools[0].invoke(keyword)
                        if debug:
                            print(f"📄 Observation: {str(result)[:150]}")
                        messages.append(HumanMessage(content=f"Observation: {result}"))
                        continue
                except Exception as e:
                    if debug:
                        print(f"⚠️ Action Input 解析失敗: {e}")

            if debug:
                print(f"✅ Final: {ai_msg.content[:200]}")
            return ai_msg.content

        return "代理人推理次數過多"

    return agent_executor
