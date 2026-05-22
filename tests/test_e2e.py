"""
End-to-end integration tests for JARVIS pipeline.

覆蓋範圍：
  - Happy path: text mode, voice mode
  - Error path: 空輸入、ASR 失敗、Brain 失敗
  - Stress path: 連續多輪對話、記憶體穩定性
  - Benchmark: token/s

前置條件：
  - Qwen1.5 MNN 模型已下載（models_qwen1.5/）
  - Whisper.cpp 已編譯（dev/whisper.cpp/build/bin/whisper-cli）
  - ffmpeg 已安裝

使用方式：
  python3 tests/test_e2e.py
  python3 -m pytest tests/test_e2e.py -v          # pytest 模式
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import time
import tracemalloc
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.jarvis_pipeline import JarvisPipeline, PipelineConfig, PipelineResult

# ============================================================================
# 輔助工具
# ============================================================================

SAMPLE_RATE = 16000


def generate_test_wav(duration_seconds: float = 2.0) -> bytes:
    """產生測試用無聲 WAV bytes。"""
    n_frames = int(SAMPLE_RATE * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def estimate_tokens(text: str) -> int:
    """粗略估算中英文混合 token 數（~2 chars/token for mixed）。"""
    return max(len(text) // 2, 1)


async def create_pipeline() -> JarvisPipeline:
    """建立並初始化 pipeline（每次測試獨立實例）。"""
    config = PipelineConfig(
        brain_max_tokens=128,
        asr_language="zh",
        tts_enabled=False,
    )
    pipeline = JarvisPipeline(config)
    await pipeline.initialize()
    return pipeline


# ============================================================================
# 測試案例
# ============================================================================

def test_text_mode_happy_path():
    """TC-01: 文字輸入 → LLM 回應（happy path）"""
    async def run():
        p = await create_pipeline()
        try:
            result = await p.run_text("台灣最高的山是什麼？")
            assert result.is_ok, f"Pipeline failed: {result.error}"
            assert result.response is not None
            assert len(result.response) > 0
            return result
        finally:
            await p.close()

    result = asyncio.run(run())
    assert "玉山" in result.response or "阿里山" in result.response, \
        f"Unexpected response: {result.response[:100]}"
    print(f"  ✅ 回應內容: {result.response[:80]}...")


def test_text_mode_empty_input():
    """TC-02: 空文字輸入（error path）"""
    async def run():
        p = await create_pipeline()
        try:
            result = await p.run_text("")
            if result.is_ok:
                print("  ⚠️  空文字仍產生回應（可能不是 bug）")
            return result
        finally:
            await p.close()

    result = asyncio.run(run())
    # Should not crash
    assert result is not None


def test_text_mode_multi_turn():
    """TC-03: 連續 3 輪對話 + 記憶體檢查"""
    async def run():
        p = await create_pipeline()
        try:
            tracemalloc.start()
            questions = [
                "1+1=?",
                "2+2=?",
                "3+3=?",
            ]
            for i, q in enumerate(questions):
                result = await p.run_text(q)
                assert result.is_ok, f"Turn {i} failed: {result.error}"
                print(f"  Turn {i+1}: {q} → {result.response[:60]}...")

            # Check memory delta
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / 1024 / 1024
            print(f"  Peak memory delta: {peak_mb:.1f} MB")
            return peak_mb
        finally:
            await p.close()

    peak = asyncio.run(run())
    assert peak < 500, f"Memory leak suspected: {peak:.0f} MB peak"


def test_voice_mode_happy_path():
    """TC-04: 語音輸入（無聲 WAV）→ ASR + LLM（error path 驗證隔離）"""
    async def run():
        p = await create_pipeline()
        try:
            wav = generate_test_wav(duration_seconds=2.0)
            result = await p.run_voice(wav)
            # ASR on silent WAV may return empty text → Brain may fail
            # This tests error isolation: pipeline should not crash
            if result.error:
                print(f"  ⚠️  ASR 預期失敗（無聲 WAV）: {result.error[:60]}")
            else:
                print(f"  ✅ ASR 辨識: {result.transcription}")
                print(f"     回應: {result.response[:60]}")
            return result
        finally:
            await p.close()

    result = asyncio.run(run())
    assert result is not None
    # Pipeline should not crash even with silent audio
    assert result.error is None or "ASR" in result.error or "Brain" in result.error


def test_voice_mode_empty_audio():
    """TC-05: 太短的語音輸入 → error fallback"""
    async def run():
        p = await create_pipeline()
        try:
            wav = generate_test_wav(duration_seconds=0.1)
            result = await p.run_voice(wav)
            return result
        finally:
            await p.close()

    result = asyncio.run(run())
    # Very short WAV should not crash the pipeline
    assert result is not None


def test_health_check():
    """TC-06: 健康檢查端點"""
    async def run():
        p = await create_pipeline()
        try:
            health = p.health_check()
            assert health["initialized"] is True
            assert health["brain_loaded"] is True
            return health
        finally:
            await p.close()

    health = asyncio.run(run())
    print(f"  ✅ 狀態: initialized={health['initialized']}, "
          f"brain={health['brain_loaded']}, "
          f"asr={health['asr_loaded']}")


def test_clean_restart():
    """TC-07: 初始化 → 關閉 → 再次初始化（無狀態殘留）"""
    async def run():
        # First lifecycle
        p1 = await create_pipeline()
        h1 = p1.health_check()
        await p1.close()

        # Second lifecycle
        p2 = await create_pipeline()
        h2 = p2.health_check()
        result = await p2.run_text("你好")
        await p2.close()

        return h1, h2, result

    h1, h2, result = asyncio.run(run())
    assert h1["initialized"] and h2["initialized"]
    assert result.is_ok
    print("  ✅ 重啟後 pipeline 正常運作")


def test_benchmark_tokens_per_second():
    """TC-08: LLM 推理速度 benchmark（目標 ≥ 15 tok/s）"""
    async def run():
        p = await create_pipeline()
        try:
            prompt = "用 100 個字介紹深度學習的歷史"
            t0 = time.time()
            result = await p.run_text(prompt)
            elapsed = time.time() - t0
            tokens = estimate_tokens(result.response)
            tps = tokens / elapsed
            return tps, elapsed, tokens, result.response[:80]
        finally:
            await p.close()

    tps, elapsed, tokens, snippet = asyncio.run(run())
    print(f"  ⏱️  {elapsed:.1f}s | tokens≈{tokens} | {tps:.1f} tok/s")
    assert tps >= 15, f"Too slow: {tps:.1f} tok/s (target ≥ 15)"
    print(f"  ✅ {tps:.1f} tok/s (目標 15)")


def test_app_module_import():
    """TC-09: app.py 模組可正常導入"""
    try:
        from app import app, get_pipeline
        assert app is not None
        print("  ✅ FastAPI app 導入成功")
    except Exception as e:
        assert False, f"app.py import failed: {e}"


def test_pipeline_result_serialization():
    """TC-10: PipelineResult WebSocket 序列化"""
    r = PipelineResult(transcription="你好", response="Hello", audio=None, error=None)
    msg = r.to_ws_message()
    assert msg["type"] == "response"
    assert msg["transcription"] == "你好"
    assert msg["response"] == "Hello"
    assert msg["error"] is None
    assert "audio" not in msg or msg["audio"] is None
    print("  ✅ WebSocket 訊息格式正確")


# ============================================================================
# 進階/手動測試（需口型同步模組）
# ============================================================================

def SKIP_test_hud_display():
    """需 T006（口型同步）完成後實作。"""
    raise NotImplementedError("T006 完成後驗證")


def SKIP_test_lip_sync():
    """需 T006（口型同步）完成後實作。"""
    raise NotImplementedError("T006 完成後驗證")


def test_vision_chat():
    """TC-11: Vision-Language 圖片辨識"""
    from brain.brain_engine import BrainEngine
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    engine = BrainEngine(model_path=model_path)
    try:
        engine.load()
        resp = engine.chat_with_image(
            "用一句話描述這張圖片",
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        )
        assert resp is not None and len(resp.strip()) > 0
        print(f"  ✅ 圖片描述: {resp.strip()[:80]}...")
    finally:
        engine.release()


# ============================================================================
# 主程式
# ============================================================================

def run_tests():
    tests = [
        ("TC-01 文字模式 happy path", test_text_mode_happy_path),
        ("TC-02 空文字輸入", test_text_mode_empty_input),
        ("TC-03 多輪對話 + 記憶體", test_text_mode_multi_turn),
        ("TC-04 語音輸入（無聲 WAV）", test_voice_mode_happy_path),
        ("TC-05 太短語音", test_voice_mode_empty_audio),
        ("TC-06 健康檢查", test_health_check),
        ("TC-07 重啟測試", test_clean_restart),
        ("TC-08 推理速度 benchmark", test_benchmark_tokens_per_second),
        ("TC-09 app.py 導入", test_app_module_import),
        ("TC-10 序列化", test_pipeline_result_serialization),
        ("TC-11 Vision-Language", test_vision_chat),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            print(f"  ✅ PASS")
            passed += 1
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"結果: {passed} passed, {failed} failed / {len(tests)}")
    print(f"{'='*50}")
    return failed == 0


if __name__ == "__main__":
    run_tests()
