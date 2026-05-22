"""Test script for MNN-LLM Brain Engine."""
import sys
import os

# add parent dir to path so "from brain.brain_engine import BrainEngine" works
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from brain.brain_engine import BrainEngine

# Also fix path for import
sys.path.insert(0, os.path.dirname(__file__))


def test_engine_load():
    """測試 1：模型載入"""
    print("=== Test 1: 引擎初始化與載入 ===")
    engine = BrainEngine()
    print(f"  model_path: {engine.model_path}")
    try:
        engine.load()
        print("  ✅ 模型載入成功")
        return engine  # ← 重要：回傳 engine 實例供後續測試使用
    except FileNotFoundError as e:
        print(f"  ⚠️  模型未找到（預期行為，請先執行 T002）")
        print(f"     {e}")
        return None
    except Exception as e:
        print(f"  ❌ 錯誤：{e}")
        return None


def test_chat(engine):
    """測試 2：對話功能"""
    print("\n=== Test 2: 對話功能 ===")
    prompts = [
        "哈囉，請用繁體中文自我介紹",
        "1+1等於多少？",
    ]
    for prompt in prompts:
        print(f"\n  Q: {prompt}")
        try:
            resp = engine.chat(prompt)
            print(f"  A: {resp[:300]}")
        except Exception as e:
            print(f"  ❌ 錯誤：{e}")


def test_generate(engine):
    """測試 3：generate() 介面"""
    print("\n=== Test 3: generate() 介面 ===")
    try:
        resp = engine.generate("說一個笑話")
        print(f"  ✅ generate() 回應：{resp[:200]}")
    except Exception as e:
        print(f"  ❌ 錯誤：{e}")


def test_tokenizer(engine):
    """測試 4：Tokenizer"""
    print("\n=== Test 4: Tokenizer ===")
    try:
        prompt = "你好"
        tokens = engine.llm.tokenizer_encode(prompt)
        decoded = engine.llm.tokenizer_decode(tokens[:5])
        print(f"  ✅ encode('{prompt}') = {tokens[:10]}...")
        print(f"     decode(tokens[:5]) = '{decoded}'")
    except Exception as e:
        print(f"  ❌ 錯誤：{e}")


def main():
    # 模型路徑：使用預設路徑（~/Projects/JARVIS-on-mac/models/）
    model_path = os.environ.get("BRAIN_MODEL_PATH", None)
    engine = BrainEngine(model_path=model_path)

    # test_engine_load() 會建立並載入 engine，回傳實例
    engine = test_engine_load()
    if engine is None:
        print("\n模型未就緒，無法繼續測試。請先完成 T002。")
        return

    test_chat(engine)
    test_generate(engine)
    test_tokenizer(engine)

    engine.release()
    print("\n✅ 全部測試完成")


if __name__ == "__main__":
    main()
