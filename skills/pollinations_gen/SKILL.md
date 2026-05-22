# 技能名稱: pollinations_gen

## 描述
使用 pollinations.ai 完全免費生成超寫實真人全身圖像，**不需要 API token**。

## 功能
- 文字生成圖像（Text-to-Image）
- 支援多种开源模型（FLUX, SDXL 等）
- 完全免費、無限使用

## 輸入參數
- `query_data`: (string) 提示詞，可選加 `模型: <model_name>`

## 支援模型
- `flux`（預設）- FLUX.1-schnell，快速高質量
- `turbo` - 更快的 FLUX 變種
- `sdxl` - Stable Diffusion XL
- `any` - 自動選擇最佳模型

## 輸出
- 返回生成的圖像（base64 PNG）

## 範例
- `A photorealistic full-body shot of a young woman in Taipei`
- `A photorealistic full-body shot... | model: flux`

## 模型代碼範例
```
寫實全身照：flux, ultra realistic, 35mm lens, natural lighting
寫實男性：sdxl, photorealistic, detailed skin texture
卡通風格：any, illustration style
```

## 技術
使用 https://image.pollinations.ai/prompt/{encoded_prompt} 直接生成，返回為圖片。