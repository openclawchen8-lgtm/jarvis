"""
TTS Voice Engine — Kokoro-82M (T005)

基於 Kokoro-82M 的文字轉語音引擎，支援中文（zf_xiaoxiao）與英文語音。
使用 PyTorch MPS（Metal Performance Shaders）在 Mac M2 上加速推理。
"""

import os
import uuid
import wave
import struct
from pathlib import Path
from typing import Optional, Generator, Tuple

import numpy as np
import soundfile as sf
import torch

# Kokoro TTS pipeline
from kokoro import KPipeline

# ============================================================================
# 設定
# ============================================================================

# 預設發音人（中文女聲）
DEFAULT_VOICE_ZH = "zf_xiaoxiao"
DEFAULT_VOICE_EN = "af_heart"

# 音頻參數
SAMPLE_RATE = 24000  # Kokoro 輸出 24kHz

# 模型路徑
VOICE_MODELS_DIR = Path(__file__).parent / "models"
KOKORO_MODEL_PATH = VOICE_MODELS_DIR / "kokoro-v1_0.pth"
VOICES_DIR = VOICE_MODELS_DIR / "voices"

# 音頻輸出目錄
AUDIO_OUTPUT_DIR = Path(__file__).parent.parent / "assets" / "audio"
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TTSEngine 類別
# ============================================================================

class TTSEngine:
    """
    Kokoro-82M TTS 引擎。

    使用方式：
        engine = TTSEngine()
        audio_path = engine.speak("你好，這是測試")
        engine.speak_streaming("這是流式輸出", callback=print)
    """

    def __init__(
        self,
        lang_code: str = "z",          # 'z'=中文, 'a'=美式英文
        voice: str = DEFAULT_VOICE_ZH,
        device: Optional[str] = None,  # None=自動偵測
        cache_dir: Optional[Path] = None,
    ):
        """
        初始化 TTS 引擎。

        Args:
            lang_code: 語言代碼。'z'=中文, 'a'=美式英文
            voice:     發音人名稱（如 'zf_xiaoxiao', 'af_heart'）
            device:     推理設備。None=自動偵測（優先 MPS，其次 CPU）
            cache_dir:  模型快取目錄。None=使用預設路徑
        """
        # 自動偵測設備
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.lang_code = lang_code
        self.voice = voice
        self.cache_dir = cache_dir or VOICE_MODELS_DIR

        # 初始化 Kokoro Pipeline
        # KPipeline 會自動下載並快取到 cache_dir
        print(f"[TTSEngine] Initializing Kokoro-82M (device={self.device}, lang={lang_code}, voice={voice})")
        self.pipeline = KPipeline(
            lang_code=lang_code,
            repo_id="hexgrad/Kokoro-82M",
            device=self.device,
        )
        print("[TTSEngine] Kokoro pipeline ready ✅")

    # ------------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------------

    def speak(self, text: str, output_path: Optional[Path] = None) -> Path:
        """
        將文字轉換為語音音頻檔案（WAV）。

        Args:
            text:        要朗讀的文字
            output_path: 輸出檔案路徑。None=自動產生

        Returns:
            輸出音頻檔案路徑（24kHz, 16-bit PCM WAV）
        """
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        # 自動產生檔案名
        if output_path is None:
            output_path = AUDIO_OUTPUT_DIR / f"tts_{uuid.uuid4().hex[:8]}.wav"

        # 合成音頻
        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(self.pipeline(text, voice=self.voice)):
            # gs  = generated sentiment (str)
            # ps  = phonemes (str)
            # audio = numpy array, dtype float32, range [-1, 1]
            audio_chunks.append(audio)

        # 合併所有 chunks
        if not audio_chunks:
            raise RuntimeError("TTS pipeline returned no audio chunks")

        full_audio = np.concatenate(audio_chunks)

        # 寫入 WAV 檔案
        self._write_wav(output_path, full_audio, SAMPLE_RATE)
        print(f"[TTSEngine] Audio saved: {output_path} ({len(full_audio)/SAMPLE_RATE:.1f}s)")
        return output_path

    def set_language(self, lang_code: str, voice: str):
        """動態切換語言/語音（重建 KPipeline）。"""
        if lang_code == self.lang_code and voice == self.voice:
            return
        self.lang_code = lang_code
        self.voice = voice
        from kokoro import KPipeline
        self.pipeline = KPipeline(
            lang_code=lang_code,
            repo_id="hexgrad/Kokoro-82M",
            device=self.device,
        )
        print(f"[TTSEngine] Switched to lang={lang_code}, voice={voice}")

    @staticmethod
    def detect_lang(text: str) -> tuple:
        """偵測文字語言，回傳 (lang_code, voice)。"""
        import re
        if re.search(r'[\u4e00-\u9fff]', text):
            return ("z", DEFAULT_VOICE_ZH)   # 中文
        return ("a", DEFAULT_VOICE_EN)        # 英文

    def speak_to_array(self, text: str) -> Tuple[np.ndarray, int]:
        """
        將文字轉換為 numpy 音頻陣列（不寫檔案）。
        自動偵測語言並切換語音。

        Args:
            text: 要朗讀的文字

        Returns:
            (audio_array, sample_rate) — audio_array 為 float32, range [-1, 1]
        """
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        # Auto-detect language and switch
        lang_code, voice = self.detect_lang(text)
        self.set_language(lang_code, voice)

        audio_chunks = []
        for gs, ps, audio in self.pipeline(text, voice=self.voice):
            audio_chunks.append(audio)

        if not audio_chunks:
            raise RuntimeError("TTS pipeline returned no audio chunks")

        return np.concatenate(audio_chunks), SAMPLE_RATE

    def speak_streaming(
        self,
        text: str,
        callback,
        chunk_seconds: float = 0.5,
    ) -> Path:
        """
        流式 TTS：每產生一個音頻 chunk 就立即回呼。

        適用於數位人口型驅動——在音頻產生的同時就能播放/驅動口型。

        Args:
            text:          要朗讀的文字
            callback:      回呼函式，簽名：callback(chunk: np.ndarray, progress: float)
                           - chunk: 當前音頻片段（float32, [-1, 1]）
                           - progress: 進度 0.0~1.0
            chunk_seconds: 回呼觸發的音頻時長（秒）。預設每 0.5 秒觸發一次

        Returns:
            完整音頻檔案路徑
        """
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        # 先估算總長度（用 pipeline 的第一個回傳）
        total_audio = []
        chunk_samples = int(SAMPLE_RATE * chunk_seconds)
        last_callback_at = 0

        for gs, ps, audio in self.pipeline(text, voice=self.voice):
            total_audio.append(audio)
            current_samples = sum(len(a) for a in total_audio)
            # 取第一個 chunk 的總長估算（不精確，但足夠給進度）
            # 更準確的做法是在 pipeline 外層估算
            progress = min(current_samples / (SAMPLE_RATE * 30), 0.99)  # 保守估計
            callback(audio, progress)

        # 合併並寫檔
        full_audio = np.concatenate(total_audio)
        output_path = AUDIO_OUTPUT_DIR / f"tts_stream_{uuid.uuid4().hex[:8]}.wav"
        self._write_wav(output_path, full_audio, SAMPLE_RATE)
        callback(np.array([]), 1.0)  # 完成信號
        return output_path

    def list_voices(self, lang_code: Optional[str] = None) -> dict:
        """
        列出可用發音人。

        Args:
            lang_code: 篩選特定語言。None=全部

        Returns:
            {lang: [voice_name, ...], ...}
        """
        # 掃描 voices 目錄
        voices_dir = VOICES_DIR
        if not voices_dir.exists():
            return {}

        voices_by_lang = {}
        for f in sorted(voices_dir.glob("*.pt")):
            name = f.stem  # e.g. "zf_xiaoxiao"
            lang = name[0]  # 'z'=中文, 'a'=英文, 'b'=英國, 'e'=其他, 'h'=混合...
            if lang_code is None or lang == lang_code:
                if lang not in voices_by_lang:
                    voices_by_lang[lang] = []
                voices_by_lang[lang].append(name)

        return voices_by_lang

    # ------------------------------------------------------------------------
    # 私有工具
    # ------------------------------------------------------------------------

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray, sample_rate: int):
        """將 float32 [-1, 1] 音頻寫入 16-bit PCM WAV。"""
        # 轉換為 16-bit integer
        audio_int = (audio * 32767).astype(np.int16)
        sf.write(str(path), audio_int, sample_rate, format="WAV", subtype="PCM_16")


# ============================================================================
# 便捷函式（可不实例化直接呼叫）
# ============================================================================

_default_engine: Optional[TTSEngine] = None


def get_engine(lang_code: str = "z", voice: str = None) -> TTSEngine:
    """取得（或建立）全域 TTSEngine 實例。"""
    global _default_engine
    if _default_engine is None:
        v = voice or (DEFAULT_VOICE_ZH if lang_code == "z" else DEFAULT_VOICE_EN)
        _default_engine = TTSEngine(lang_code=lang_code, voice=v)
    return _default_engine


def speak(text: str, output_path: Optional[Path] = None) -> Path:
    """便捷函式：使用預設引擎朗讀文字。"""
    return get_engine().speak(text, output_path)
