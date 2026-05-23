"""
Model Router — 多模型後端切換器

支援在執行期間動態切換模型來源，不需重啟伺服器。

用法：
    router = ModelRouter()
    router.use("mnn-qwen1.5")
    router.use("openai")        # 需要 OPENAI_API_KEY
    router.use("mnn-qwen3.5")

    response = router.chat("你好")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional, Protocol

logger = logging.getLogger("jarvis.router")


# ============================================================================
# 模型後端介面
# ============================================================================

class ModelBackend(Protocol):
    """模型後端必須實作的介面。"""
    name: str
    def chat(self, prompt: str) -> str: ...
    def load(self) -> bool: ...
    def release(self): ...


# ============================================================================
# MNN 本地模型後端
# ============================================================================

class MNNBackend:
    """MNN-LLM 本地推理後端。"""

    def __init__(self, model_dir: str, name: str = "mnn"):
        self.model_dir = model_dir
        self.name = name
        self._llm = None

    def load(self) -> bool:
        from MNN.llm import create
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"config.json not found: {config_path}")
            return False
        self._llm = create(config_path)
        self._llm.load()
        logger.info(f"MNNBackend 已載入: {self.name} ({self.model_dir})")
        return True

    def chat(self, prompt: str) -> str:
        if not self._llm:
            return "模型未載入"
        from brain.brain_engine import BrainEngine
        wrapped = BrainEngine._wrap_prompt_static(prompt)  # reuse prompt wrapper
        self._llm.reset()
        return self._llm.response(wrapped)

    def release(self):
        if self._llm:
            try:
                self._llm.release_module(0)
            except (AttributeError, TypeError):
                pass
            self._llm = None


# ============================================================================
# OpenAI API 後端
# ============================================================================

class OpenAIBackend:
    """OpenAI API 遠端後端（需 API key）。"""

    def __init__(self, name: str = "openai"):
        self.name = name
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def load(self) -> bool:
        if not self.api_key:
            logger.warning("OPENAI_API_KEY 未設定，OpenAI 後端無法使用")
            return False
        try:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
            logger.info(f"OpenAIBackend 就緒: {self.model}")
            return True
        except ImportError:
            logger.warning("openai 套件未安裝 (pip install openai)")
            return False

    def chat(self, prompt: str) -> str:
        if not hasattr(self, '_client') or not self._client:
            return "OpenAI 未就緒"
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"OpenAI 錯誤：{e}"

    def release(self):
        self._client = None


# ============================================================================
# Router
# ============================================================================

_PROJ_ROOT = Path(__file__).resolve().parent.parent

BACKENDS: Dict[str, str] = {
    "mnn-qwen1.5":  str(_PROJ_ROOT / "models_qwen1.5"),
    "mnn-qwen3.5":  str(_PROJ_ROOT / "models"),
}


class ModelRouter:
    """模型路由：管理多個後端，提供動態切換。"""

    def __init__(self):
        self._current: Optional[ModelBackend] = None
        self._available: Dict[str, ModelBackend] = {}
        self._init_backends()

    def _init_backends(self):
        """初始化所有可用後端。"""
        for name, path in BACKENDS.items():
            backend = MNNBackend(path, name=name)
            self._available[name] = backend

        # OpenAI (optional)
        openai_b = OpenAIBackend()
        if openai_b.load():
            self._available["openai"] = openai_b

    def list_models(self) -> Dict[str, bool]:
        """列出所有可用模型及其載入狀態。"""
        return {name: b._llm is not None for name, b in self._available.items()}

    def use(self, name: str) -> bool:
        """切換到指定模型。返回是否成功。"""
        if name not in self._available:
            logger.error(f"未知模型: {name}，可用: {list(self._available.keys())}")
            return False

        backend = self._available[name]

        # 載入（若尚未載入）
        if not hasattr(backend, '_llm') or backend._llm is None:
            if not backend.load():
                return False

        self._current = backend
        logger.info(f"切換到模型: {name}")
        return True

    def chat(self, prompt: str) -> str:
        """透過當前模型對話。"""
        if self._current is None:
            # 預設使用第一個
            first = list(self._available.keys())[0]
            self.use(first)
        return self._current.chat(prompt)

    def release_all(self):
        for b in self._available.values():
            b.release()
