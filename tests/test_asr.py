"""
測試：ASR Engine — Whisper.cpp + Metal GPU

用法：
  python3 test_asr.py           # 全部測試
  python3 test_asr.py file      # 檔案模式測試
  python3 test_asr.py mic 5    # 錄音 5 秒測試

前置條件：
  1. whisper.cpp 已編譯（build/bin/whisper-cli）
  2. Whisper GGML 模型已下載（ggml-base.bin）
  3. ffmpeg 已安裝（brew install ffmpeg）
  4. 終端機已有麥克風權限（System Preferences → Privacy → Microphone）
"""

import sys
import os

# add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.asr_engine import WhisperASR, check_dependencies

JFK_WAV = "/Users/claw/Projects/JARVIS-on-mac/whisper.cpp/samples/jfk.wav"


def test_file_mode():
    """測試 1：檔案轉寫（英文）"""
    print("\n[Test 1] 檔案模式 — jfk.wav")
    check_dependencies()
    asr = WhisperASR(language="zh")  # 會自動偵測（jfk 是英文）
    result = asr.benchmark(JFK_WAV)
    assert "text" in result, f"Benchmark failed: {result}"
    assert len(result["text"]) > 0, "Empty transcription"
    print(f"  ✅ Time: {result['elapsed_seconds']}s | Text: {result['text'][:80]}")
    return result


def test_mic_mode(duration=3.0):
    """測試 2：麥克風錄音轉寫"""
    print(f"\n[Test 2] 麥克風模式 — {duration}s 錄音")
    asr = WhisperASR(language="zh")
    try:
        text = asr.transcribe_mic(duration=duration)
        print(f"  ✅ Text: {text[:100]}")
    except Exception as e:
        print(f"  ⚠️  麥克風測試失敗：{e}")
        print("  → 確保終端機已有麥克風權限")


def test_multiple_models():
    """測試 3：多模型支援"""
    print("\n[Test 3] 多模型支援")
    models = {
        "base (多語言)": "/Users/claw/Projects/whisper.cpp/models/ggml-base.bin",
        "base.en (英文)": "/Users/claw/Projects/whisper.cpp/models/ggml-base.en.bin",
    }
    for name, path in models.items():
        if os.path.exists(path):
            asr = WhisperASR(model_path=path, language="zh")
            result = asr.benchmark(JFK_WAV)
            print(f"  {name}: {result['elapsed_seconds']}s → {result['text'][:50]}")
        else:
            print(f"  ⚠️  {name} not found")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "file":
            test_file_mode()
        elif sys.argv[1] == "mic":
            duration = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
            test_mic_mode(duration)
        else:
            print("Usage: test_asr.py [file|mic]")
    else:
        print("=== WhisperASR 測試套件 ===")
        test_file_mode()
        test_multiple_models()
        test_mic_mode(duration=3.0)
