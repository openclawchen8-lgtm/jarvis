"""
TTS Engine Test — test_tts.py

用法：
    cd ~/Projects/JARVIS-on-mac
    /opt/homebrew/bin/python3.11 -m pytest tests/test_tts.py -v

    # 或直接執行：
    /opt/homebrew/bin/python3.11 tests/test_tts.py
"""

import sys
import time
from pathlib import Path

# 確保專案根目錄在 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice.voice_engine import TTSEngine, speak, get_engine
import torch
import numpy as np


def test_tts_engine_import():
    """測試 TTSEngine 可以正確引入。"""
    from voice.voice_engine import TTSEngine
    assert TTSEngine is not None


def test_device_detection():
    """測試設備偵測（MPS / CPU）。"""
    if torch.cuda.is_available():
        expected = "cuda"
    elif torch.backends.mps.is_available():
        expected = "mps"
    else:
        expected = "cpu"

    engine = TTSEngine()
    assert engine.device == expected, f"Expected {expected}, got {engine.device}"
    print(f"✅ Device: {engine.device}")


def test_list_voices():
    """測試發音人列表。"""
    engine = TTSEngine()
    voices = engine.list_voices()
    assert "z" in voices, "Chinese voices (z*) should be available"
    assert "zf_xiaoxiao" in voices["z"], "zf_xiaoxiao should be available"
    print(f"✅ Available voices: {voices}")


def test_speak_chinese():
    """測試中文 TTS。"""
    engine = TTSEngine(lang_code="z", voice="zf_xiaoxiao")
    text = "你好，JARVIS 語音合成測試成功！"

    start = time.time()
    path = engine.speak(text)
    elapsed = time.time() - start

    assert path.exists(), f"Output file not created: {path}"
    assert path.suffix == ".wav", f"Expected .wav, got {path.suffix}"
    size_kb = path.stat().st_size / 1024
    print(f"✅ Chinese TTS: {path} ({size_kb:.0f}KB, {elapsed:.1f}s)")


def test_speak_english():
    """測試英文 TTS。"""
    engine = TTSEngine(lang_code="a", voice="af_heart")
    text = "Hello, this is JARVIS voice synthesis test."

    start = time.time()
    path = engine.speak(text)
    elapsed = time.time() - start

    assert path.exists(), f"Output file not created: {path}"
    size_kb = path.stat().st_size / 1024
    print(f"✅ English TTS: {path} ({size_kb:.0f}KB, {elapsed:.1f}s)")


def test_speak_to_array():
    """測試 speak_to_array（不回寫檔案）。"""
    engine = TTSEngine(lang_code="z", voice="zf_xiaoxiao")
    text = "測試 numpy 音頻輸出"

    audio, sr = engine.speak_to_array(text)
    assert isinstance(audio, np.ndarray), "Should return numpy array"
    assert audio.dtype == np.float32, "Should be float32"
    assert sr == 24000, "Sample rate should be 24000"
    assert len(audio) > 0, "Audio should not be empty"
    print(f"✅ speak_to_array: {len(audio)/sr:.1f}s audio, shape={audio.shape}")


def test_streaming_callback():
    """測試流式 TTS 回呼。"""
    engine = TTSEngine(lang_code="z", voice="zf_xiaoxiao")
    text = "這是流式輸出測試"
    chunks = []
    total_samples = 0

    def callback(chunk: np.ndarray, progress: float):
        nonlocal total_samples
        chunks.append(len(chunk))
        total_samples += len(chunk)

    path = engine.speak_streaming(text, callback=callback)

    assert path.exists(), "Output file should be created"
    assert len(chunks) > 0, "Should have received at least one chunk"
    print(f"✅ Streaming: {len(chunks)} chunks, {total_samples/24000:.1f}s audio")


def test_invalid_text():
    """測試空文字處理。"""
    engine = TTSEngine()
    try:
        engine.speak("")
        assert False, "Should raise ValueError for empty text"
    except ValueError as e:
        assert "empty" in str(e).lower()
        print(f"✅ Empty text handled correctly: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("TTS Engine Tests — Kokoro-82M")
    print("=" * 60)

    test_device_detection()
    test_list_voices()
    test_speak_chinese()
    test_speak_english()
    test_speak_to_array()
    test_streaming_callback()
    test_invalid_text()

    print("=" * 60)
    print("✅ All TTS tests passed!")
    print("=" * 60)
