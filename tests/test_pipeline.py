"""
End-to-end pipeline test.

測試方式：
  python3 -m pytest tests/test_pipeline.py -v
  python3 tests/test_pipeline.py          # 直接執行
"""

import asyncio
import io
import os
import sys
import wave
import time

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.jarvis_pipeline import JarvisPipeline, PipelineConfig, PipelineResult


# ============================================================================
# 工具
# ============================================================================

def generate_test_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """產生測試用 WAV bytes（無聲或簡單正弦波）。"""
    n_channels = 1
    n_frames = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        # 寫入無聲 frames
        w.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


# ============================================================================
# 測試
# ============================================================================

def test_pipeline_import():
    """測試 1：模組可正常導入。"""
    from pipeline.jarvis_pipeline import JarvisPipeline, PipelineConfig
    from pipeline import JarvisPipeline as JP
    print("  ✅ 模組導入成功")


def test_config():
    """測試 2：PipelineConfig 預設值。"""
    config = PipelineConfig()
    assert config.asr_language == "zh", f"Expected zh, got {config.asr_language}"
    assert config.voice_output is False
    assert config.brain_max_tokens == 256
    print("  ✅ PipelineConfig 預設值正確")


def test_result_to_ws_message():
    """測試 3：PipelineResult 序列化。"""
    result = PipelineResult(
        transcription="你好",
        response="你好！我是 JARVIS。",
        audio=None,
        error=None,
    )
    msg = result.to_ws_message()
    assert msg["type"] == "response"
    assert msg["transcription"] == "你好"
    assert msg["response"] == "你好！我是 JARVIS。"
    assert msg["error"] is None
    print("  ✅ PipelineResult.to_ws_message() 正確")


def test_result_error():
    """測試 4：PipelineResult 錯誤處理。"""
    result = PipelineResult(error="Something went wrong")
    assert result.is_ok is False
    assert result.error == "Something went wrong"
    msg = result.to_ws_message()
    assert msg["error"] == "Something went wrong"
    print("  ✅ PipelineResult 錯誤處理正確")


def test_wav_generation():
    """測試 5：WAV 產生工具。"""
    wav = generate_test_wav(duration_seconds=1.0)
    assert len(wav) > 1000, "WAV too short"
    # 驗證 WAV header
    buf = io.BytesIO(wav)
    with wave.open(buf, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getsampwidth() == 2
    print(f"  ✅ WAV 產生正確 ({len(wav)} bytes)")


async def test_pipeline_text_mode():
    """測試 6：Pipeline text mode（實際串聯）。"""
    print("  ⏳ 初始化 Pipeline...")
    config = PipelineConfig(brain_max_tokens=128)
    pipeline = JarvisPipeline(config)
    await pipeline.initialize()

    # 健康檢查
    health = pipeline.health_check()
    assert health["initialized"] is True, f"Brain not initialized: {health}"
    assert health["brain_loaded"] is True, "Brain not loaded"
    print(f"  ✅ Pipeline 健康檢查通過: {health}")

    # Text mode
    print("  ⏳ Text mode 測試（可能需要幾十秒）...")
    result = await pipeline.run_text("台灣最高的山是什麼？")
    print(f"  ✅ Text mode 回應: {result.response[:100] if result.response else 'None'}")
    assert result.is_ok, f"Pipeline failed: {result.error}"

    await pipeline.close()


async def test_pipeline_health():
    """測試 7：Pipeline 健康檢查（不實際呼叫大腦）。"""
    config = PipelineConfig()
    pipeline = JarvisPipeline(config)
    health = pipeline.health_check()
    assert health["initialized"] is False
    assert health["brain_loaded"] is False
    print("  ✅ Pipeline 初始狀態正確（未初始化）")


# ============================================================================
# 主程式
# ============================================================================

def run_tests():
    print("=" * 60)
    print("JARVIS Pipeline 測試套件")
    print("=" * 60)

    # 同步測試
    tests = [
        test_pipeline_import,
        test_config,
        test_result_to_ws_message,
        test_result_error,
        test_wav_generation,
    ]

    print("\n[同步測試]")
    for test in tests:
        name = test.__name__
        try:
            test()
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback; traceback.print_exc()

    # 非同步測試
    print("\n[非同步測試]")
    async_tests = [
        test_pipeline_health,
        test_pipeline_text_mode,
    ]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for test in async_tests:
        name = test.__name__
        try:
            loop.run_until_complete(test())
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    print("測試完成")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
