"""
JARVIS Voice Module — 語音辨識 + 語音合成

子模組：
  - asr_engine: Whisper.cpp ASR（語音→文字）
  - voice_engine: Kokoro-82M TTS（文字→語音）
"""

from voice.asr_engine import WhisperASR, ASRError, check_dependencies

__all__ = ["WhisperASR", "ASRError", "check_dependencies"]
