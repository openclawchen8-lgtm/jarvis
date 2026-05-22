"""MNN-LLM Brain Engine - JARVIS core intelligence."""
import json
import os
from MNN.llm import create, LlmStatus


class BrainEngine:
    """MNN-LLM 大腦引擎封裝。

    使用方式：
        brain = BrainEngine(model_path="path/to/model")
        brain.load()
        response = brain.chat("你好，幫我叫醒豪")
        brain.release()
    """

    # 模型切換：設環境變數 BRAIN_MODEL
    #   models/         → Qwen3.5-0.8B（較強較慢）
    #   models_qwen1.5/ → Qwen1.5-0.5B（較小較快，預設）
    _DEFAULT_MODEL = os.environ.get(
        "BRAIN_MODEL",
        "models_qwen1.5/"  # 預設 Qwen1.5
    )

    # Prompt 模板（各模型格式不同）
    _PROMPT_TEMPLATES = {
        "Qwen2":   "{{system}}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "Qwen3.5": "{{system}}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "default": "{{system}}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    }

    def __init__(self, model_path: str = None):
        """初始化引擎。

        Args:
            model_path: MNN 量化模型目錄（需含 config.json）。
                       預設值：由 BRAIN_MODEL 環境變數控制（預設 models_qwen1.5/）
        """
        if model_path is None:
            default = self._DEFAULT_MODEL
            if not default.startswith("/"):
                default = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), default
                )
            model_path = default
        self.model_path = model_path
        self.llm = None
        self._loaded = False
        self._model_type = None  # 延遲偵測

    def _detect_model_type(self) -> str:
        """從 config.json 偵測模型類型（Qwen2 / Qwen3.5）。"""
        config_path = os.path.join(self.model_path, "config.json")
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            mtype = cfg.get("model_type", "default")
            print(f"[BrainEngine] 偵測到 model_type: {mtype}")
            return mtype
        except Exception as e:
            print(f"[BrainEngine] 偵測 model_type 失敗 ({e})，使用 default")
            return "default"

    def _wrap_prompt(self, prompt: str, system_prompt: str = "") -> str:
        """根據模型類型自動包裝 prompt，支援 system prompt。"""
        if self._model_type is None:
            self._model_type = self._detect_model_type()
        template = self._PROMPT_TEMPLATES.get(
            self._model_type,
            self._PROMPT_TEMPLATES["default"]
        )
        system_part = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
        return template.replace("{{system}}", system_part).format(prompt=prompt)

    def _get_config_path(self) -> str:
        """取得 config.json 路徑（MNN.create() 需要 config.json 而非目錄）。"""
        config = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config):
            raise FileNotFoundError(
                f"[BrainEngine] config.json 未找到：{config}\n"
                f"請執行 scripts/download_qwen1.5_model.sh 或 scripts/download_qwen_model.sh"
            )
        return config

    def load(self) -> bool:
        """載入模型（同步，會 block）。"""
        if self._loaded:
            print("[BrainEngine] 模型已載入，跳過")
            return True
        config_path = self._get_config_path()
        print(f"[BrainEngine] 載入模型：{self.model_path}")
        print(f"[BrainEngine] config.json：{config_path}")
        self.llm = create(config_path)  # ← 重要：傳 config.json 而非目錄
        self.llm.load()
        self._loaded = True
        self._model_type = self._detect_model_type()
        print("[BrainEngine] 模型載入完成 ✅")
        return True

    def chat(self, prompt: str, max_tokens: int = 256, system_prompt: str = "") -> str:
        """單輪對話（blocking）。自動套用模型專屬 prompt 格式。"""
        if not self._loaded:
            raise RuntimeError("[BrainEngine] 請先呼叫 brain.load()")
        self.llm.reset()
        full_prompt = self._inject_context(prompt)
        wrapped = self._wrap_prompt(full_prompt, system_prompt)
        resp = self.llm.response(wrapped)
        return resp

    def _inject_context(self, prompt: str) -> str:
        """透過技能系統處理使用者輸入，注入上下文。"""
        from skills import SkillRegistry
        from skills.builtin import date_time_skill, weather_skill
        registry = SkillRegistry()
        registry.register(date_time_skill)
        registry.register(weather_skill)
        result, matched = registry.process(prompt)
        if matched:
            return result
        return f"當前時間：{self._now_str()}\n使用者問：{prompt}"

    @staticmethod
    def _now_str() -> str:
        import datetime
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        now = datetime.datetime.now()
        return f"{now.year}年{now.month}月{now.day}日 {weekdays[now.weekday()]}"

    def chat_with_image(self, prompt: str, image_path: str) -> str:
        """多模態對話：文字 + 圖片。

        支援 Qwen3.5-VL 格式：
          <img>image_path</img>prompt

        Args:
            prompt: 文字提問
            image_path: 圖片路徑（本地或 URL）

        Returns:
            模型回應文字
        """
        if not self._loaded:
            raise RuntimeError("[BrainEngine] 請先呼叫 brain.load()")
        self.llm.reset()
        full_prompt = f"<img>{image_path}</img>{prompt}"
        wrapped = self._wrap_prompt(full_prompt)
        resp = self.llm.response(wrapped)
        return resp

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """generate() 介面（與 chat 等價）。"""
        return self.chat(prompt, max_tokens)

    def release(self):
        """釋放模型資源。"""
        if self.llm:
            try:
                self.llm.release_module(0)
            except (AttributeError, TypeError):
                pass  # GC 會自動清理
            self.llm = None
            self._loaded = False
            print("[BrainEngine] 資源已釋放")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"[BrainEngine Test] model_path={model_path}")
    engine = BrainEngine(model_path=model_path)
    try:
        engine.load()
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("請執行 scripts/download_qwen1.5_model.sh 或 scripts/download_qwen_model.sh")
        sys.exit(1)
    test_prompt = "你好，請用繁體中文自我介紹"
    print(f"[Test] Prompt: {test_prompt}")
    response = engine.chat(test_prompt)
    print(f"[BrainEngine] Response:\n{response}")
    engine.release()
    print("✅ BrainEngine 測試通過")
