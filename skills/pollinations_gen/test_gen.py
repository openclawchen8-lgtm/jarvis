#!/usr/bin/env python3
"""驗證 pollinations_gen skill"""

import sys
import os

# 手動加入專案根目錄
sys.path.insert(0, os.path.expanduser("~/Projects/JARVIS-on-mac"))

from skills.pollinations_gen.execute import execute

def test():
    prompt = "A photorealistic full-body shot of a young woman standing in Taipei, head-to-toe view, 35mm lens"
    print(f"Testing pollinations generation...")
    print(f"Prompt: {prompt}")

    result = execute(prompt)

    if "[IMAGE_DATA]:" in result:
        b64 = result.split("[IMAGE_DATA]:")[1]
        print(f"✅ SUCCESS: Image generated, base64 length = {len(b64)}")
        # 儲存測試圖
        import base64
        data = base64.b64decode(b64)
        out_path = os.path.expanduser("~/Projects/JARVIS-on-mac/test_pollinations.png")
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"✅ Saved to {out_path}")
        return True
    else:
        print(f"❌ FAILED: {result}")
        return False

if __name__ == "__main__":
    test()