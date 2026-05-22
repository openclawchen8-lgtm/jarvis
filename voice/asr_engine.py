"""
ASR Engine — Whisper.cpp (Metal GPU accelerated) + ffmpeg recording.

語音辨識引擎，使用 whisper.cpp 的 whisper-cli 配合 Metal GPU 加速，
ffmpeg 處理麥克風錄音（avfoundation），支援中文（普通話）+ 英文。

架構：
  ffmpeg (錄音) → WAV (16kHz mono) → whisper-cli (推論) → 文字

依賴：
  - whisper.cpp (已編譯)：build/bin/whisper-cli
  - ffmpeg (已安裝)：音頻錄製
  - Whisper GGML 模型：ggml-base.bin（多語言）或 ggml-base.en.bin（英文）
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

# ============================================================================
# 設定
# ============================================================================

WHISPER_CLI = os.environ.get(
    "WHISPER_CLI",
    "/Users/claw/Projects/JARVIS-on-mac/dev/whisper.cpp/build/bin/whisper-cli"
)

DEFAULT_MODEL = os.environ.get(
    "WHISPER_MODEL",
    "/Users/claw/Projects/JARVIS-on-mac/dev/whisper.cpp/models/ggml-small.bin"
)

SAMPLE_RATE = 16000
CHANNELS = 1


# ============================================================================
# 例外
# ============================================================================

class ASRError(Exception):
    """ASR 引擎錯誤。"""
    pass


# ============================================================================
# 檢查依賴
# ============================================================================

def check_dependencies():
    """檢查依賴是否可用。"""
    import shutil
    errors = []
    if not os.path.exists(WHISPER_CLI):
        errors.append(f"whisper-cli not found: {WHISPER_CLI}")
    if not shutil.which("ffmpeg"):
        errors.append("ffmpeg not found (brew install ffmpeg)")
    if not os.path.exists(DEFAULT_MODEL):
        errors.append(f"Whisper model not found: {DEFAULT_MODEL}")
    if errors:
        raise ASRError("Missing dependencies:\n" + "\n".join(errors))


# ============================================================================
# ASR 引擎
# ============================================================================

class WhisperASR:
    """
    Whisper ASR 引擎（whisper.cpp + Metal GPU）。

    用法：
        asr = WhisperASR()
        text = asr.transcribe("audio.wav")
        text = asr.transcribe_mic(duration=5)  # 錄音 5 秒後辨識
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        language: str = "zh",
        threads: int = 4,
        whisper_cli: Optional[str] = None,
    ):
        self.model_path = model_path or DEFAULT_MODEL
        self.language = language
        self.threads = threads
        self.whisper_cli = whisper_cli or WHISPER_CLI
        self._validate()

    def _validate(self):
        """驗證依賴。"""
        if not os.path.exists(self.whisper_cli):
            raise ASRError(f"whisper-cli not found: {self.whisper_cli}")
        if not os.path.exists(self.model_path):
            raise ASRError(f"model not found: {self.model_path}")

    def transcribe(self, audio_path: str) -> str:
        """
        轉寫音頻檔案（支援 .wav, .mp3, .ogg, .flac）。

        Args:
            audio_path: 音頻檔路徑

        Returns:
            辨識文字（已去除空白）

        Raises:
            ASRError: 轉寫失敗時
        """
        if not os.path.exists(audio_path):
            raise ASRError(f"Audio file not found: {audio_path}")

        cmd = [
            self.whisper_cli,
            "-m", self.model_path,
            "-f", audio_path,
            "-l", self.language,
            "-nt",          # no timestamps
            "-t", str(self.threads),
            "-fa",          # Flash Attention (Metal GPU)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            raise ASRError("Transcription timed out (>60s)")

        if result.returncode != 0:
            stderr = "\n".join(
                line for line in result.stderr.split("\n")
                if line.strip() and not any(
                    line.startswith(x) for x in
                    ("ggml_", "whisper_init", "system_info", "whisper_backend", "whisper_model", "whisper_print")
                )
            )
            raise ASRError(f"whisper-cli failed:\n{stderr}")

        # 解析輸出
        lines = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line and not line.startswith((
                "whisper_", "system_info", "main:",
                "loading", "processing", "progress"
            )):
                lines.append(line)

        return " ".join(lines).strip()

    def transcribe_mic(self, duration: float = 5.0) -> str:
        """
        從麥克風錄音並轉寫。

        Args:
            duration: 錄音時長（秒）

        Returns:
            辨識文字
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "avfoundation",
                "-i", ":0",
                "-ar", str(SAMPLE_RATE),
                "-ac", str(CHANNELS),
                "-t", str(duration),
                "-loglevel", "error",
                wav_path,
            ]

            ffmpeg_result = subprocess.run(
                cmd, capture_output=True, timeout=duration + 5
            )
            if ffmpeg_result.returncode != 0:
                raise ASRError(
                    f"ffmpeg recording failed: {ffmpeg_result.stderr.decode()}"
                )

            return self.transcribe(wav_path)

        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def benchmark(self, audio_path: Optional[str] = None) -> dict:
        """
        效能基準測試。

        Args:
            audio_path: 測試音頻路徑（預設用 jfk.wav）

        Returns:
            包含耗時和結果的 dict
        """
        if audio_path is None:
            audio_path = "/Users/claw/Projects/JARVIS-on-mac/whisper.cpp/samples/jfk.wav"

        if not os.path.exists(audio_path):
            return {"error": f"Test audio not found: {audio_path}"}

        start = time.time()
        text = self.transcribe(audio_path)
        elapsed = time.time() - start

        return {
            "elapsed_seconds": round(elapsed, 2),
            "text": text,
            "model": os.path.basename(self.model_path),
            "language": self.language,
        }


# ============================================================================
# CLI 入口
# ============================================================================

if __name__ == "__main__":
    import sys
    print("=== WhisperASR Test ===")
    check_dependencies()

    asr = WhisperASR(language="zh")
    print(f"Model: {os.path.basename(asr.model_path)}")
    print(f"Whisper CLI: {asr.whisper_cli}")

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Transcribing: {audio_file}")
        text = asr.transcribe(audio_file)
        print(f"Result: {text}")
    else:
        print("Recording 3 seconds from microphone...")
        text = asr.transcribe_mic(duration=3.0)
        print(f"Result: {text}")
        print("\n=== Benchmark ===")
        result = asr.benchmark()
        print(result)
