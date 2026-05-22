# 技能名稱: hf_image_gen

## 描述
使用 Hugging Face API 生成超寫實真人全身圖像。

## 功能模式

### 模式一：FLUX.1 文字生成圖（預設）
直接透過文字描述生成圖像。
- 支援 FLUX.1-schnell（快速，4 步驟）
- 支援 FLUX.1-dev（高品質）

### 模式二：InstantID 臉部參考
給定一張人臉範例圖，保持同樣的人臉生成新的全身圖。
- 使用 InstantX/InstantID 模型
- 可調整 identity_scale 控制人臉相似度（0.0-1.0）

### 模式三：Outpainting 影像延伸
對現有圖像進行外擴延伸（泳池風格等）。
- 需要提供參考圖片路徑
- 支援向下或四周延伸

## 輸入參數
- `query_data`: (string) 包含提示詞和模式指定

### 格式：
- FLUX 生成：`<提示詞>` 或 `FLUX: <提示詞>`
- InstantID：`InstantID: <提示詞> | face: <圖片路徑>`
- Outpainting：`Outpaint: <提示詞> | image: <圖片路徑>`

## 輸出
- 返回生成的圖像（base64 PNG）

## 所需設定
在 `~/.jarvis_config.json` 中設定：
```json
{
  "hf_image_gen": {
    "api_token": "你的_HF_TOKEN",
    "default_model": "schnell"
  }
}
```

取得 HF Token：https://huggingface.co/settings/tokens

## 範例
- `FLUX: A photorealistic full-body shot of a young woman...`
- `InstantID: A photorealistic full-body shot... | face: /path/to/face.jpg`
- `Outpaint: crystal clear pool, summer day... | image: /path/to/halfbody.jpg`