# JARVIS-on-mac

> 本機離線 AI 助理，MacBook Air M2 專用，打造鋼鐵人 JARVIS 風格的私人智慧助手。

## 快速開始

```bash
# 1. 一鍵安裝所有相依（Homebrew + Python + 模型）
bash scripts/setup.sh

# 2. 啟動伺服器
bash scripts/jarvis.sh start

# 3. 開啟瀏覽器
open http://localhost:8000
```

### 依賴清單

- **Python 3.11**（Homebrew）— 專用於 JARVIS
- **系統套件**：ffmpeg、cmake、ninja
- **Python 套件**：見 `requirements.txt`（FastAPI、PyTorch、ONNX、MNN、Whisper、Kokoro 等）
- **模型**：
  - Qwen LLM（`scripts/download_qwen_model.sh`）
  - Whisper ASR（`scripts/download_whisper_model.sh`）
  - LiteAvatar 2D 數位人（`bash renderers/liteavatar/download_models.sh` + 頭像下載）

### 頭像下載

LiteAvatar 2D 數位人頭像從 ModelScope Gallery 下載：

```bash
# 瀏覽頭像
open https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery/summary

# 下載單個頭像（ID 從瀏覽器網址列複製）
bash renderers/liteavatar/download_avatar.sh <頭像ID> <目錄名稱>

# 範例
bash renderers/liteavatar/download_avatar.sh 20250408/P1wRwMpa9BBZa1d5O9qiAsCw woman1

# 列出已下載頭像
bash renderers/liteavatar/download_avatar.sh --list
```

### 管理指令

```bash
bash scripts/jarvis.sh start    # 啟動 Web + Telegram
bash scripts/jarvis.sh stop     # 關閉
bash scripts/jarvis.sh status   # 檢查健康狀態
bash scripts/jarvis.sh restart  # 重啟
```

## 專案結構

```
JARVIS-on-mac/
├── brain/            # MNN-LLM 大腦引擎
├── voice/            # 語音合成（TTS）+ 語音辨識（ASR）
├── render/           # 數位人驅動（口型同步）
├── app.py            # FastAPI + WebSocket 主程式
├── tests/            # 單元測試
└── assets/           # Avatar 素材、模型權重
```

## 開發日誌（How-To）

每個實作步驟皆同步記錄於 `~/howto/`：

| Howto | 對應任務 | 內容 |
|-------|---------|------|
| [build-mnn-metal.md](https://github.com/openclawchen8-lgtm/openclaw-howto/blob/main/build-mnn-metal.md) | T001 | MNN 編譯環境建置（Metal 加速） |
| [qwen-model-download.md](../howto/qwen-model-download.md) | T002 | Python Binding 安裝 + Qwen 模型下載 |
| [javis-pipeline.md](../howto/javis-pipeline.md) | T003 | 大腦 + 語音 + 數位人串接 |
| [whisper-cpp-integration.md](../howto/whisper-cpp-integration.md) | T004 | Whisper.cpp 語音轉文字整合 |
| [kokoro-tts-integration.md](../howto/kokoro-tts-integration.md) | T005 | Kokoro-82M TTS 語音合成整合 ✅ |
| [frontend-hud-design.md](../howto/frontend-hud-design.md) | T005-FE | 前端 HUD 設計（待建） |
| [digital-human-validation.md](../howto/digital-human-validation.md) | T006 | 口型同步測試（待建） |
| [integration-testing.md](../howto/integration-testing.md) | T007 | 端到端整合測試（待建） |

## 架構

```
User Input (Voice/Text)
    ↓
┌─────────────────────┐
│  ASR (sherpa-mnn)   │  ← 語音轉文字
└─────────────────────┘
    ↓
┌─────────────────────┐
│  LLM (Qwen2.5 MNN)  │  ← 大腦推理
└─────────────────────┘
    ↓
┌─────────────────────┐
│  TTS (bert-vits2)   │  ← 語音合成
└─────────────────────┘
    ↓
┌─────────────────────┐
│  A2BS (UniTalker)   │  ← 口型驅動
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Render (NNR/其他)  │  ← 數位人影片輸出
└─────────────────────┘
```

## 技術棧

- **推理引擎**：MNN（alibaba/MNN）Metal 加速
- **LLM**：Qwen2.5-7B-Instruct（MNN 量化版）
- **TTS**：Kokoro-82M + bert-vits2-MNN
- **ASR**：sherpa-mnn-streaming
- **數位人**：UniTalker-MNN + KlingAI LivePortrait
- **通訊**：FastAPI + WebSocket（對內）+ REST API（對外）
- **硬體**：MacBook Air M2 16GB，記憶體峰值 < 10GB

## 里程碑

| 階段 | 內容 | 狀態 |
|------|------|------|
| Phase 1 | 環境建置（MNN Metal） | 🔄 T001-T002 |
| Phase 2 | 後端流水線整合 | 📋 T003-T005 |
| Phase 3 | 前端 HUD + 整合測試 | 📋 T004, T006-T007 |

## 專案位置

原始碼：`~/Projects/JARVIS-on-mac/`
任務追蹤：`~/Tasks/Javis/`
How-To 文件：`~/howto/`

### 模型下載腳本

```bash
# Qwen3.5-0.8B（較強較慢）
bash scripts/download_qwen_model.sh

# Qwen1.5-0.5B（較小較快，預設）
bash scripts/download_qwen1.5_model.sh
```
```
