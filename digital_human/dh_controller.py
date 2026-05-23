"""
Digital Human Controller — 統一調度 TTS → 數位人管線 (T043)

功能：
  - 根據 DHConfig.mode 決定走 BODY / FACE / NONE 路徑
  - TTS 輸出音訊 + viseme + blendshape → 封裝為前端可消費的格式
  - LLM 動作指令解析（body mode 的 cmd:xxx）
  - 健康檢查與降級邏輯

目前 BODY mode 已完工（T042），FACE mode 待 T044 完成後接入。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from digital_human.dh_config import DHConfig, DigitalHumanMode, MODE_LABELS

logger = logging.getLogger("jarvis.dh")


class DigitalHumanController:
    """數位人控制器 — 統一臉部/身體模式的生命週期與路由。"""

    def __init__(self, config: Optional[DHConfig] = None):
        self.config = config or DHConfig()
        self._initialized = False
        self._tts_engine = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        """初始化 DH 相關資源。"""
        if self._initialized:
            return True

        if self.config.mode == DigitalHumanMode.NONE:
            logger.info("DH: 已停用")
            self._initialized = True
            return True

        if not self.config.is_available():
            logger.warning(f"DH: 模式 {self.config.mode.value} 無法使用（依賴不滿足）")
            return False

        # Init TTS (shared across modes)
        if self.config.tts_enabled:
            try:
                from voice.voice_engine import get_engine
                self._tts_engine = get_engine()
                logger.info("DH: TTS 引擎就緒 ✅")
            except Exception as e:
                logger.warning(f"DH: TTS 初始化失敗: {e}")
                self.config.tts_enabled = False

        mode_label = MODE_LABELS.get(self.config.mode, "unknown")
        logger.info(f"DH: 模式 [{mode_label}] 初始化完成 ✅")
        self._initialized = True
        return True

    def close(self):
        """釋放資源。"""
        self._initialized = False
        logger.info("DH: 已關閉")

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Core: Process TTS output for DH display
    # ------------------------------------------------------------------

    def process_tts_result(
        self, audio_arr: np.ndarray, sr: int, text: str
    ) -> dict:
        """將 TTS 輸出處理為 DH 前端可消費的格式。

        Args:
            audio_arr: float32 [-1, 1] audio
            sr: sample rate
            text: 原始文字（用於 LLM 動作指令解析）

        Returns:
            dict with keys for frontend rendering:
            - audio: base64 WAV
            - viseme_track: viseme frames
            - blendshape: A2BS blendshape (if enabled)
            - command: LLM action command (body mode)
        """
        result = {
            "audio": None,
            "viseme_track": None,
            "blendshape": None,
            "command": None,
        }

        if not self._initialized:
            return result

        # 1. Audio → base64 WAV
        import base64 as b64
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((audio_arr * 32767).astype(np.int16).tobytes())
        result["audio"] = b64.b64encode(buf.getvalue()).decode()

        # 2. Viseme track (energy-based)
        if self.config.viseme_enabled:
            try:
                from render.render_engine import get_engine
                engine = get_engine(fps=30)
                track = engine.render_from_audio(audio_arr, sr)
                viseme = engine.render_to_json(track)
                frames = viseme.get("frames", [])
                openness = [f["o"] for f in frames]
                logger.debug(
                    f"DH viseme: {len(frames)} frames, "
                    f"openness [{min(openness):.2f}–{max(openness):.2f}]"
                )
                result["viseme_track"] = viseme
            except Exception as e:
                logger.warning(f"DH viseme 生成失敗: {e}")

        # 3. A2BS blendshape (ML-based, if enabled)
        if self.config.a2bs_enabled:
            try:
                from pipeline.a2bs_engine import A2BSEngine as _A2BS
                engine = _A2BS()
                if engine.load():
                    bs = engine.process(audio_arr, sr)
                    if bs and bs.get("num_frames", 0) > 0:
                        result["blendshape"] = {
                            "type": "blendshape",
                            "fps": bs["fps"],
                            "num_frames": bs["num_frames"],
                            "coeffs": bs["coeffs"],
                        }
            except Exception as e:
                logger.warning(f"DH A2BS 失敗: {e}")

        # 4. LLM action command (body mode)
        if self.config.mode == DigitalHumanMode.BODY and self.config.auto_command:
            result["command"] = self._parse_command(text)

        return result

    # ------------------------------------------------------------------
    # TTS convenience
    # ------------------------------------------------------------------

    def speak_to_array(self, text: str):
        """Run TTS, return (audio_arr, sr)."""
        if not self._tts_engine:
            raise RuntimeError("TTS not initialized")
        return self._tts_engine.speak_to_array(text)

    # ------------------------------------------------------------------
    # Command parsing
    # ------------------------------------------------------------------

    _COMMANDS = {"idle", "walk", "lie_down", "prone", "talking"}

    @staticmethod
    def _parse_command(text: str) -> Optional[str]:
        """Extract cmd:xxx from LLM response text."""
        if not text:
            return None
        import re
        m = re.search(r'cmd:(\w+)', text)
        if m and m.group(1) in DigitalHumanController._COMMANDS:
            return m.group(1)
        return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return {
            "initialized": self._initialized,
            **self.config.health(),
        }
