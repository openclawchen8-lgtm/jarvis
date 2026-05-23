"""
JARVIS Pipeline — 流水線協調器

整合 ASR（Whisper）+ 大腦（BrainEngine）+ TTS（Kokoro）
支援 text-in / voice-in 兩種輸入模式。

訊息格式（WebSocket）：
  輸入（text mode）：
    {"type": "text", "content": "你好，JARVIS"}

  輸入（voice mode）：
    {"type": "voice"}

  輸出：
    {
      "type": "response",
      "transcription": "你好，JARVIS",  # 語音模式的辨識結果
      "response": "你好！我有什麼可以幫你？",
      "audio": "<base64 WAV>",          # voice-out mode才有
      "error": null
    }
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# 關掉 Kokoro/PyTorch deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="kokoro")

logger = logging.getLogger("jarvis.pipeline")

# ============================================================================
# 組態
# ============================================================================

@dataclass
class PipelineConfig:
    """Pipeline 設定。"""
    # 大腦設定
    brain_model_path: Optional[str] = None          # None = 預設 models/
    brain_max_tokens: int = 256
    brain_temperature: float = 0.7

    # ASR 設定
    asr_language: str = "zh"
    asr_model_path: Optional[str] = None

    # TTS 設定
    tts_enabled: bool = True
    tts_voice: str = "zf_xiaoxiao"  # Kokoro 中文女聲
    tts_lang_code: str = "z"        # 'z'=中文, 'a'=英文

    # 行為設定
    voice_output: bool = True    # True = 回 WAV，False = 回文字
    stream_chunks: bool = True   # True = 流式輸出
    system_prompt: str = "你是 JARVIS，一個 AI 助理，用繁體中文回答。"

    # A2BS (Audio-to-BlendShape) 設定
    a2bs_enabled: bool = True    # True = 啟用 UniTalker A2BS

    # Digital Human (T043) 設定
    dh_mode: str = "none"        # "none" / "body" / "face" / "hybrid"
    dh_auto_command: bool = True # LLM 自動判斷動作指令


# ============================================================================
# Pipeline
# ============================================================================

class JarvisPipeline:
    """
    JARVIS 核心流水線。

    用法：
        config = PipelineConfig()
        pipeline = JarvisPipeline(config)
        await pipeline.initialize()

        # Text mode
        response = await pipeline.run_text("你好，JARVIS")

        # Voice mode
        response = await pipeline.run_voice(wav_bytes)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._brain = None
        self._asr = None
        self._tts = None
        self._a2bs = None
        self._dh = None
        self._initialized = False

    # ------------------------------------------------------------------------
    # 生命週期
    # ------------------------------------------------------------------------

    async def initialize(self) -> None:
        """初始化所有模組（非同步，避免阻塞）。"""
        if self._initialized:
            return

        logger.info("初始化 Pipeline...")

        # 載入 BrainEngine（最慢，需非同步）
        loop = asyncio.get_event_loop()
        self._brain = await loop.run_in_executor(None, self._init_brain)

        # 載入 ASR
        self._asr = self._init_asr()

        # TTS
        if self.config.tts_enabled:
            self._tts = self._init_tts()

        # A2BS
        if self.config.a2bs_enabled:
            self._a2bs = self._init_a2bs()

        # Digital Human (T043)
        self._dh = self._init_dh()

        self._initialized = True
        logger.info("Pipeline 初始化完成 ✅")

    def _init_brain(self):
        """同步初始化大腦（在 executor 中執行）。"""
        from brain.brain_engine import BrainEngine
        brain = BrainEngine(model_path=self.config.brain_model_path)
        brain.load()  # 重要：實際載入 MNN 模型
        return brain

    def _init_asr(self):
        """初始化 ASR。"""
        from voice.asr_engine import WhisperASR
        return WhisperASR(
            model_path=self.config.asr_model_path,
            language=self.config.asr_language,
        )

    def _init_tts(self):
        """初始化 TTS（若相依套件不存在則優雅降級）。"""
        try:
            from voice.voice_engine import TTSEngine
            return TTSEngine(
                lang_code=self.config.tts_lang_code,
                voice=self.config.tts_voice,
            )
        except ImportError as e:
            logger.warning(f"TTS 初始化失敗（{e}），TTS 功能已停用")
            self.config.tts_enabled = False
            return None

    def _init_dh(self):
        """初始化 Digital Human 控制器。"""
        if self.config.dh_mode in ("", "none"):
            return None
        try:
            from digital_human import DHConfig, DigitalHumanMode, DigitalHumanController
            mode = {
                "body": DigitalHumanMode.BODY,
                "face": DigitalHumanMode.FACE,
                "hybrid": DigitalHumanMode.HYBRID,
            }.get(self.config.dh_mode, DigitalHumanMode.NONE)
            dconfig = DHConfig(
                mode=mode,
                tts_enabled=self.config.tts_enabled,
                viseme_enabled=True,
                a2bs_enabled=self.config.a2bs_enabled,
                auto_command=self.config.dh_auto_command,
            )
            ctrl = DigitalHumanController(dconfig)
            loop = asyncio.get_event_loop()
            ok = loop.run_until_complete(ctrl.initialize())
            if ok:
                logger.info(f"DH 控制器初始化完成 ✅ (mode={self.config.dh_mode})")
                return ctrl
            else:
                logger.warning(f"DH 控制器初始化失敗 (mode={self.config.dh_mode})")
                return None
        except Exception as e:
            logger.warning(f"DH 控制器初始化異常: {e}")
            return None

    def _init_a2bs(self):
        """初始化 A2BS (UniTalker-MNN)。若失敗則優雅降級。"""
        try:
            from pipeline.a2bs_engine import A2BSEngine
            engine = A2BSEngine()
            if engine.load():
                logger.info("A2BS 引擎初始化完成 ✅")
                return engine
            else:
                logger.warning("A2BS 載入失敗，已降級")
                self.config.a2bs_enabled = False
                return None
        except Exception as e:
            logger.warning(f"A2BS 初始化失敗（{e}），已降級")
            self.config.a2bs_enabled = False
            return None

    async def close(self) -> None:
        """釋放資源。"""
        if self._brain is not None:
            await asyncio.get_event_loop().run_in_executor(
                None, self._brain.release
            )
        if self._a2bs is not None:
            self._a2bs.close()
        if self._dh is not None:
            self._dh.close()
        self._initialized = False
        logger.info("Pipeline 已關閉")

    # ------------------------------------------------------------------------
    # 主要 API
    # ------------------------------------------------------------------------

    async def run_text(self, text: str) -> PipelineResult:
        """
        Text mode：文字輸入 → 大腦 → 回應文字。

        Args:
            text: 使用者輸入文字

        Returns:
            PipelineResult（含大腦回應）
        """
        if not self._initialized:
            await self.initialize()

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._brain.chat, text, 256, self.config.system_prompt
            )
            command, response = _parse_command(response)
            dh_command = None
            audio = viseme_track = blendshape = None
            if self.config.voice_output and self._tts is not None:
                audio, viseme_track, blendshape = await self._tts_step(response)
                # DH controller may override command
                if self._dh and self._dh.is_loaded:
                    from digital_human import DigitalHumanMode
                    if self._dh.config.mode == DigitalHumanMode.BODY:
                        cmd_from_dh = self._dh._parse_command(response)
                        if cmd_from_dh:
                            dh_command = cmd_from_dh
            final_command = dh_command or command
            return PipelineResult(
                transcription=text,
                response=response,
                audio=audio,
                viseme_track=viseme_track,
                blendshape=blendshape,
                command=final_command,
                error=None,
            )
        except Exception as e:
            logger.error(f"Brain error: {e}")
            return PipelineResult(
                transcription=text,
                response=None,
                audio=None,
                error=str(e),
            )

    async def run_voice(self, wav_data: bytes) -> PipelineResult:
        """
        Voice mode：WAV 音頻 → ASR → 大腦 → 回應。

        Args:
            wav_data: 音頻資料（bytes）

        Returns:
            PipelineResult（含辨識文字 + 回應）
        """
        if not self._initialized:
            await self.initialize()

        transcription = None
        response_text = None
        audio = None
        error = None

        # Step 1: ASR（語音→文字）
        try:
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                None, self._asr_transcribe, wav_data
            )
            logger.info(f"ASR: {transcription}")
        except Exception as e:
            logger.error(f"ASR error: {e}")
            error = f"ASR 失敗：{e}"
            return PipelineResult(
                transcription=None,
                response=None,
                audio=None,
                error=error,
            )

        # Step 2: Brain（文字→回應）
        try:
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None, self._brain.chat, transcription, 256, self.config.system_prompt
            )
            command, response_text = _parse_command(response_text)
            logger.info(f"Brain: {response_text[:80]}")
        except Exception as e:
            logger.error(f"Brain error: {e}")
            error = f"大腦失敗：{e}"
            return PipelineResult(
                transcription=transcription,
                response=None,
                audio=None,
                error=error,
            )

        # Step 3: TTS（回應→語音 + viseme track + blendshape）
        audio = viseme_track = blendshape = None
        dh_command = None
        if self.config.voice_output and self._tts is not None:
            audio, viseme_track, blendshape = await self._tts_step(response_text)
            if self._dh and self._dh.is_loaded:
                from digital_human import DigitalHumanMode
                if self._dh.config.mode == DigitalHumanMode.BODY:
                    cmd = self._dh._parse_command(response_text)
                    if cmd:
                        dh_command = cmd
        final_command = dh_command or command

        return PipelineResult(
            transcription=transcription,
            response=response_text,
            audio=audio,
            viseme_track=viseme_track,
            blendshape=blendshape,
            command=final_command,
            error=None,
        )

    async def _tts_step(self, text: str) -> Tuple[Optional[bytes], Optional[dict], Optional[dict]]:
        """
        TTS 步驟：文字 → (WAV bytes, viseme_track dict, blendshape dict)。

        若 DH 模式啟用，路由至 DigitalHumanController 處理。
        失敗不回拋，回 (None, None, None)。
        """
        try:
            loop = asyncio.get_event_loop()
            audio_arr, sr = await loop.run_in_executor(
                None, self._tts.speak_to_array, text
            )

            # DH 模式：透過控制器統一處理
            if self._dh is not None and self._dh.is_loaded:
                dh_result = await loop.run_in_executor(
                    None, self._dh.process_tts_result, audio_arr, sr, text
                )
                audio_b64 = dh_result.get("audio")
                viseme_track = dh_result.get("viseme_track")
                blendshape = dh_result.get("blendshape")

                import base64
                audio_bytes = base64.b64decode(audio_b64) if audio_b64 else None

                if viseme_track:
                    frames = viseme_track.get("frames", [])
                    logger.info(f"DH VISEME: {len(frames)} frames")
                if blendshape:
                    logger.info(f"DH A2BS: {blendshape['num_frames']} frames @ {blendshape['fps']}fps")

                # Merge command into pipeline result
                cmd = dh_result.get("command")
                if cmd:
                    logger.info(f"DH command: {cmd}")

                return audio_bytes, viseme_track, blendshape

            # 非 DH 模式：原有邏輯
            import io
            import wave
            import numpy as np

            viseme_track = None
            try:
                from render.render_engine import get_engine
                engine = get_engine(fps=30)
                track = engine.render_from_audio(audio_arr, sr)
                viseme_track = engine.render_to_json(track)
                frames = viseme_track.get("frames", [])
                openness_vals = [f["o"] for f in frames]
                logger.info(f"VISEME: {len(frames)} frames, openness range [{min(openness_vals):.2f}-{max(openness_vals):.2f}], "
                           f"blinks: {len(viseme_track.get('blinks', []))}")
            except Exception as ve:
                logger.warning(f"Viseme generation failed (non-fatal): {ve}")

            blendshape = None
            if self.config.a2bs_enabled and self._a2bs is not None:
                try:
                    result = await loop.run_in_executor(
                        None, self._a2bs.process, audio_arr, sr
                    )
                    if result and result["num_frames"] > 0:
                        blendshape = {
                            "type": "blendshape",
                            "fps": result["fps"],
                            "num_frames": result["num_frames"],
                            "coeffs": result["coeffs"],
                        }
                        bs_track = _blendshape_to_viseme(result, duration=len(audio_arr)/sr)
                        if bs_track:
                            viseme_track = bs_track
                            frames = viseme_track.get("frames", [])
                            openness_vals = [f["o"] for f in frames]
                            logger.info(f"A2BS VISEME: {len(frames)} frames, openness range [{min(openness_vals):.2f}-{max(openness_vals):.2f}]")
                        logger.info(f"A2BS: {result['num_frames']} frames @ {result['fps']}fps")
                except Exception as ae:
                    logger.warning(f"A2BS failed (non-fatal): {ae}")

            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes((audio_arr * 32767).astype(np.int16).tobytes())
            return buf.getvalue(), viseme_track, blendshape
        except Exception as e:
            logger.warning(f"TTS failed (non-fatal): {e}")
            return None, None, None

    def _asr_transcribe(self, wav_data: bytes) -> str:
        """
        將 WAV bytes 寫入暫存檔，呼叫 WhisperASR 轉寫。
        """
        import tempfile
        from voice.asr_engine import WhisperASR

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_data)
            wav_path = f.name

        try:
            asr = WhisperASR(
                model_path=self.config.asr_model_path,
                language=self.config.asr_language,
            )
            return asr.transcribe(wav_path)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    # ------------------------------------------------------------------------
    # 批次 / 工具
    # ------------------------------------------------------------------------

    def health_check(self) -> dict:
        """健康檢查。"""
        dh_status = None
        if self._dh is not None:
            dh_status = self._dh.health()
        return {
            "initialized": self._initialized,
            "brain_loaded": self._brain is not None,
            "asr_loaded": self._asr is not None,
            "tts_loaded": self._tts is not None,
            "a2bs_loaded": self._a2bs is not None,
            "dh_mode": self.config.dh_mode,
            "dh_loaded": self._dh is not None and self._dh.is_loaded,
            "voice_output": self.config.voice_output,
        }


# ============================================================================
# 工具函式
# ============================================================================

def _blendshape_to_viseme(
    bs_result: dict, duration: float, target_fps: int = 30
) -> Optional[dict]:
    """
    Convert A2BS blendshape coefficients (53-d: 50 expr + 3 jaw_pose)
    to frontend-compatible viseme_track format.

    jaw_pose is [rx, ry, rz] in radians; rz ≈ jaw opening (negative = open).
    """
    import numpy as np
    coeffs = np.array(bs_result["coeffs"], dtype=np.float32)
    if coeffs.ndim != 2 or coeffs.shape[1] < 53:
        return None

    expr = coeffs[:, :50]
    jaw = coeffs[:, 50:]  # [rx, ry, rz]

    # Derive openness from jaw rz (map radians [-1, 0.5] → [0, 1])
    raw_open = -jaw[:, 1]  # rz
    open_min, open_max = -0.5, 0.8
    openness = np.clip((raw_open - open_min) / (open_max - open_min), 0.0, 1.0)

    # Derive intensity from expression magnitude
    intensity = np.clip(np.mean(np.abs(expr), axis=1) * 3.0, 0.0, 1.0)

    # Map to 12 viseme IDs based on openness + expression shape
    openness_to_viseme = [0, 2, 4, 5, 8, 3, 7, 1]  # REST→B→D→E→I→C→G→A
    n_buckets = len(openness_to_viseme)

    # Resample from BS fps to target_fps
    n_in = len(coeffs)
    n_out = max(1, int(duration * target_fps))

    x_old = np.linspace(0, n_in - 1, n_in)
    x_new = np.linspace(0, n_in - 1, n_out)
    openness_r = np.interp(x_new, x_old, openness)
    intensity_r = np.interp(x_new, x_old, intensity)

    frames = []
    for i in range(n_out):
        o = float(round(float(openness_r[i]), 2))
        bucket = min(int(o * n_buckets), n_buckets - 1)
        vid = openness_to_viseme[bucket]
        frames.append({
            "v": vid,
            "t": round(i / target_fps, 3),
            "i": float(round(float(intensity_r[i]), 2)),
            "o": o,
            "r": 0.5 if vid in (9, 10, 11) else 0.0,
        })

    return {
        "type": "viseme_track",
        "fps": target_fps,
        "duration": round(duration, 2),
        "has_audio": True,
        "blinks": [],
        "frames": frames,
    }

_COMMANDS = {"idle", "walk", "lie_down", "prone", "talking"}

def _parse_command(text: str) -> tuple:
    """
    從大腦回應中提取動作指令。
    指令格式：回應文字中的首個 cmd:xxx 關鍵字。
    若無則回傳 (None, text)。
    """
    if not text:
        return None, text
    import re
    m = re.search(r'cmd:(\w+)', text)
    if m and m.group(1) in _COMMANDS:
        cleaned = re.sub(r'cmd:\w+\s*', '', text).strip()
        return m.group(1), cleaned
    return None, text


# ============================================================================
# 結果結構
# ============================================================================

@dataclass
class PipelineResult:
    """Pipeline 執行結果。"""
    transcription: Optional[str] = None   # 語音辨識結果（voice mode）
    response: Optional[str] = None          # 大腦回應
    audio: Optional[bytes] = None           # TTS 音頻（base64 編碼）
    viseme_track: Optional[dict] = None     # viseme 動畫時間軸（energy-based）
    blendshape: Optional[dict] = None       # blendshape 係數（UniTalker A2BS）
    command: Optional[str] = None           # 動作指令（idle|walk|lie_down|prone|talking）
    error: Optional[str] = None             # 錯誤訊息

    def to_ws_message(self) -> dict:
        """轉為 WebSocket JSON 訊息。"""
        msg = {
            "type": "response",
            "transcription": self.transcription,
            "response": self.response,
            "error": self.error,
        }
        if self.audio is not None:
            msg["audio"] = base64.b64encode(self.audio).decode()
        if self.viseme_track is not None:
            msg["viseme_track"] = self.viseme_track
        if self.blendshape is not None:
            msg["blendshape"] = self.blendshape
        if self.command is not None:
            msg["command"] = self.command
        return msg

    @property
    def is_ok(self) -> bool:
        return self.error is None and self.response is not None
