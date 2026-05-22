"""
JARVIS FastAPI 應用程式

WebSocket 端點：ws://host/ws/chat
HTTP 端點：
  GET  /          → 歡迎頁
  GET  /health    → 健康檢查
  WS   /ws/chat   → JARVIS 對話

訊息格式（客戶端 → 伺服器）：

  Text mode:
    {"type": "text", "content": "你好，JARVIS"}

  Voice mode:
    {"type": "voice", "data": "<base64 WAV>"}
    或直接發送 binary WAV

訊息格式（伺服器 → 客戶端）：

  {"type": "response", "transcription": "...", "response": "...", "error": null}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

# 關掉 PyTorch deprecation warnings（Kokoro TTS 用舊 API）
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ============================================================================
# 日誌
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jarvis.app")

from pipeline.jarvis_pipeline import PipelineResult


# ============================================================================
# Pipeline 全域實例
# ============================================================================

import asyncio

_pipeline = None
_pipeline_lock = asyncio.Lock()


async def get_pipeline():
    """取得全域 Pipeline 實例（lazy init，執行緒安全）。"""
    global _pipeline
    if _pipeline is None:
        async with _pipeline_lock:
            if _pipeline is None:
                from pipeline.jarvis_pipeline import JarvisPipeline, PipelineConfig, PipelineResult
                config = PipelineConfig()
                # 從設定檔讀取 system_prompt
                config_path = Path.home() / ".jarvis_config.json"
                if config_path.exists():
                    try:
                        with open(config_path, encoding="utf-8") as f:
                            root = json.load(f)
                        sp = root.get("system_prompt", "").strip()
                        if sp:
                            config.system_prompt = sp
                    except Exception:
                        pass
                _pipeline = JarvisPipeline(config)
                await _pipeline.initialize()
    return _pipeline

# ============================================================================
# 模型設定（讀取 models.json）
# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理。"""
    # Startup
    logger.info("JARVIS 啟動中...")

    # 讀取上次使用的模型（只對本地 MNN 模型有效）
    config_path = Path.home() / ".jarvis_config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                root = json.load(f)
            last = root.get("_last_model", "")
            cfgs = _load_models_config()["backends"]
            if last in cfgs and cfgs[last].get("type") == "mnn":
                os.environ["BRAIN_MODEL"] = cfgs[last].get("model_dir", "models_qwen1.5") + "/"
                logger.info(f"恢復上次模型: {last}")
        except Exception as e:
            logger.warning(f"無法讀取設定檔恢复模型: {e}")

    pipeline = await get_pipeline()
    logger.info("JARVIS 已就緒 ✅")

    # Pre-load LiteAvatar engine
    try:
        avatars_dir = Path(__file__).parent / "renderers" / "liteavatar" / "avatars"
        if avatars_dir.exists():
            avatars = sorted([d for d in avatars_dir.iterdir() if d.is_dir() and (d / "neutral_pose.npy").exists()])
            if avatars:
                logger.info(f"LiteAvatar: 預載頭像 {avatars[0].name}...")
                await _liteavatar_ensure_engine(str(avatars[0]), avatars[0].name)
    except Exception as e:
        logger.warning(f"LiteAvatar 預載失敗: {e}")

    yield

    # Shutdown
    logger.info("JARVIS 關閉中...")
    if _pipeline is not None:
        await _pipeline.close()


app = FastAPI(
    title="JARVIS API",
    description="JARVIS 語音助手 WebSocket API",
    version="0.1.0",
    lifespan=lifespan,
)

# 安全性設定
MAX_WAV_SIZE = 10 * 1024 * 1024  # 10MB
API_KEY = os.environ.get("JARVIS_API_KEY", "")

# CORS 允許的域名（從環境變數讀取，逗號分隔）
CORS_ORIGINS = os.environ.get("JARVIS_CORS_ORIGINS", "*").split(",")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 簡易 API 認證裝飾器
def require_api_key():
    async def check_key(api_key: str = None):
        if not API_KEY:
            return  # 未設定 API key 時略過
        if api_key != API_KEY:
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="Invalid API key")
    return check_key

# 靜態檔案（圖表、頭像等）
assets_dir = Path(__file__).parent / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


# ============================================================================
# 模型設定（讀取 models.json）
# ============================================================================

def _load_models_config() -> dict:
    """讀取 ~/.jarvis_config.json 中的 models 設定。
    過濾：不存在的 MNN 模型目錄、無 API key 的 OpenAI 模型。"""
    config_path = Path.home() / ".jarvis_config.json"
    if not config_path.exists():
        return {"backends": {}}
    with open(config_path, encoding="utf-8") as f:
        root = json.load(f)
    project_root = Path(__file__).parent
    backends = {}
    for key, val in root.get("models", {}).get("backends", {}).items():
        if val.get("enabled") is False:
            continue
        if val.get("type") == "openai" and not val.get("api_key"):
            continue
        # MNN 模型：檢查目錄是否存在且有模型檔
        if val.get("type") == "mnn":
            model_dir = val.get("model_dir", "")
            abs_dir = project_root / model_dir if model_dir else None
            if not abs_dir or not abs_dir.is_dir():
                continue
            if not any(abs_dir.glob("*.mnn")):
                continue
        backends[key] = val
    return {"backends": backends}


async def _openai_chat(content: str, model_key: str):
    """透過 OpenAI 相容 API 對話（支援 function calling agent loop）。"""
    cfgs = _load_models_config()["backends"]
    cfg = cfgs.get(model_key)
    if not cfg or cfg.get("type") != "openai":
        return PipelineResult(transcription=content, response=None, audio=None, error=f"未知模型: {model_key}")
    try:
        # 讀取技能定義
        from skills.agent_loop import chat_with_tools, load_skills
        skills = load_skills(Path.home() / ".jarvis_config.json")

        reply = await chat_with_tools(
            content=content,
            model_key=model_key,
            api_key=cfg["api_key"],
            api_base=cfg.get("api_base", "https://zenmux.ai/api/v1"),
            model=cfg["model"],
            skills=skills,
            system_prompt=cfg.get("system_prompt", ""),
        )
        # OpenAI 回覆也跑 TTS
        audio = None
        try:
            from voice.voice_engine import get_engine
            tts = get_engine()
            arr, sr = tts.speak_to_array(reply)
            import io, wave, numpy as np
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((arr * 32767).astype(np.int16).tobytes())
            audio = buf.getvalue()
        except Exception as e:
            logger.warning(f"TTS 生成失敗: {e}")
        return PipelineResult(transcription=content, response=reply, audio=audio, error=None)
    except Exception as e:
        logger.error(f"OpenAI 錯誤: {e}")
        return PipelineResult(transcription=content, response=None, audio=None, error=str(e))


# ============================================================================
# HTTP 端點
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """HUD 主頁。"""
    hud_path = Path(__file__).parent / "assets" / "hud.html"
    if hud_path.exists():
        return HTMLResponse(hud_path.read_text(encoding="utf-8"))
    return """
    <!DOCTYPE html>
    <html><head><title>JARVIS</title></head>
    <body>
      <h1>🤖 JARVIS API</h1>
      <p>WebSocket: <code>ws://host/ws/chat</code></p>
      <p>健康檢查: <a href="/health">/health</a></p>
      <p>HUD 前端: <a href="/hud">/hud</a></p>
    </body>
    </html>
    """

@app.get("/hud", response_class=HTMLResponse)
async def hud():
    """HUD 前端頁面。"""
    hud_path = Path(__file__).parent / "assets" / "hud.html"
    if hud_path.exists():
        return HTMLResponse(hud_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>HUD 頁面未找到</h1>", status_code=404)

@app.get("/call", response_class=HTMLResponse)
async def call_page(room: str = ""):
    """語音通話頁面（WebRTC 直接通話）。"""
    html = """<!DOCTYPE html>
<html lang="zh-TW">
<head><meta charset="UTF-8"><title>JARVIS 通話</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e17;color:#c0d0e0;font-family:'Courier New',monospace;height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px}
.status{font-size:14px;color:#00d4ff;letter-spacing:2px}
.btn{background:transparent;border:1px solid #00d4ff;color:#00d4ff;padding:14px 32px;border-radius:8px;cursor:pointer;font-size:16px;font-family:inherit;transition:all 0.3s}
.btn:hover{background:rgba(0,180,255,0.15)}
.btn.active{background:rgba(255,51,85,0.2);border-color:#ff3355;color:#ff3355;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.room{font-size:12px;color:rgba(192,208,224,0.4)}
.response{max-width:500px;text-align:center;min-height:40px;padding:10px 20px;border:1px solid rgba(0,180,255,0.3);border-radius:8px;display:none}
.response.show{display:block}
</style></head>
<body>
<div class="room">""" + room + """</div>
<div class="status" id="status">● 待命</div>
<button class="btn" id="micBtn">🎤 按下說話</button>
<div class="response" id="response"></div>
<script>
const ws = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/ws/chat");
let mediaRecorder = null, audioChunks = [], isRecording = false;
const btn = document.getElementById("micBtn");
const statusEl = document.getElementById("status");
const respEl = document.getElementById("response");

ws.onmessage = e => {
  const d = JSON.parse(e.data);
  if (d.error) { respEl.textContent = "⚠ " + d.error; respEl.classList.add("show"); return; }
  if (d.transcription) respEl.textContent = "👤 " + d.transcription;
  if (d.response) { respEl.textContent = "🤖 " + d.response; respEl.classList.add("show"); }
  if (d.audio) { new Audio("data:audio/wav;base64," + d.audio).play(); }
  statusEl.textContent = "● 待命";
};

btn.onclick = async () => {
  if (isRecording) { mediaRecorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    mediaRecorder = new MediaRecorder(stream, {mimeType:'audio/webm'});
    audioChunks = []; isRecording = true;
    btn.classList.add("active"); btn.textContent = "🔴 放開";
    statusEl.textContent = "● 錄音中...";
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      isRecording = false; btn.classList.remove("active"); btn.textContent = "🎤 按下說話";
      stream.getTracks().forEach(t => t.stop());
      if (!audioChunks.length) return;
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuf = await ctx.decodeAudioData(await new Blob(audioChunks,{type:'audio/webm'}).arrayBuffer());
      const sr = 16000, len = Math.floor(audioBuf.length * sr / audioBuf.sampleRate);
      const data = new Float32Array(len);
      for (let i = 0; i < len; i++) {
        let s = 0;
        for (let c = 0; c < audioBuf.numberOfChannels; c++) s += audioBuf.getChannelData(c)[Math.floor(i * audioBuf.length / len)];
        data[i] = Math.max(-1, Math.min(1, s / audioBuf.numberOfChannels));
      }
      const wav = new ArrayBuffer(44 + data.length * 2);
      const dv = new DataView(wav);
      const w = (o,v)=>dv.setUint16(o,v,true), W=(o,v)=>dv.setUint32(o,v,true);
      new TextEncoder().encodeInto("RIFF", new Uint8Array(wav));
      W(4, wav.byteLength - 8); new TextEncoder().encodeInto("WAVE", new Uint8Array(wav,8));
      new TextEncoder().encodeInto("fmt ", new Uint8Array(wav,12)); W(16, 16); w(20,1); w(22,1); W(24, sr); W(28, sr*2); w(32,2); w(34,16);
      new TextEncoder().encodeInto("data", new Uint8Array(wav,36)); W(40, data.length * 2);
      for (let i = 0; i < data.length; i++) dv.setInt16(44 + i*2, Math.max(-32768, Math.min(32767, data[i] * 32767)), true);
      ws.send(wav);
      statusEl.textContent = "● 處理中...";
    };
    mediaRecorder.start();
  } catch(e) { respEl.textContent = "⚠ " + e.message; respEl.classList.add("show"); }
};
</script>
</body></html>"""
    return HTMLResponse(html)

@app.get("/health")
async def health():
    """健康檢查端點。"""
    pipeline = await get_pipeline()
    status = pipeline.health_check()
    return {
        "status": "ok" if status["initialized"] else "initializing",
        **status,
    }

@app.get("/models")
async def list_models():
    """列出可用模型。"""
    cfgs = _load_models_config()
    current_key = os.environ.get("BRAIN_MODEL", "mnn-qwen1.5").rstrip("/")
    # 把目錄路徑轉回 model key
    for k, v in cfgs["backends"].items():
        if v.get("model_dir", "").rstrip("/") == current_key or k == current_key:
            current_key = k
            break
    available = {k: v.get("label", k) for k, v in cfgs["backends"].items()}
    return {"available": available, "current": current_key}

@app.post("/switch-model")
async def switch_model(data: dict):
    """切換模型（設定寫入 ~/.jarvis_config.json，重啟後保留）。"""
    model = data.get("model", "")
    cfgs = _load_models_config()
    if model not in cfgs["backends"]:
        return {"type": "error", "error": f"未知模型: {model}，可用: {list(cfgs['backends'].keys())}"}
    cfg = cfgs["backends"][model]
    if cfg["type"] == "mnn":
        os.environ["BRAIN_MODEL"] = cfg.get("model_dir", "models_qwen1.5") + "/"
    else:
        os.environ["BRAIN_MODEL"] = model

    # 寫入設定檔，重啟後保留
    config_path = Path.home() / ".jarvis_config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                root = json.load(f)
            root["_last_model"] = model
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"無法寫入設定檔: {e}")

    # MNN 模型需重啟 pipeline
    if cfg["type"] == "mnn":
        global _pipeline
        if _pipeline is not None:
            await _pipeline.close()
            _pipeline = None
    return {"status": "ok", "model": model, "type": cfg["type"]}


# ============================================================================
# LiteAvatar 端點（選用，需先下載模型）
# ============================================================================

_liteavatar_engine = None
_liteavatar_current_avatar = None
_liteavatar_loading = False
_liteavatar_lock = asyncio.Lock()

def _liteavatar_load_sync(avatar_dir: str) -> tuple:
    """Load LiteAvatar engine synchronously (runs in executor)."""
    from renderers.liteavatar.engine import LiteAvatarEngine
    engine = LiteAvatarEngine(avatar_dir)
    ok = engine.load()
    return (engine, ok)

async def _liteavatar_ensure_engine(avatar_dir: str, avatar_id: str) -> bool:
    """Ensure LiteAvatar engine is loaded for the given avatar."""
    global _liteavatar_engine, _liteavatar_current_avatar, _liteavatar_loading
    
    async with _liteavatar_lock:
        if _liteavatar_engine is not None and _liteavatar_current_avatar == avatar_id and _liteavatar_engine.is_loaded:
            return True
        
        if _liteavatar_engine is not None:
            _liteavatar_engine.close()
            _liteavatar_engine = None
        
        loop = asyncio.get_event_loop()
        _liteavatar_loading = True
        try:
            engine, ok = await loop.run_in_executor(None, _liteavatar_load_sync, avatar_dir)
            if ok:
                _liteavatar_engine = engine
                _liteavatar_current_avatar = avatar_id
                return True
            return False
        finally:
            _liteavatar_loading = False

# LiteAvatar pre-load is handled in lifespan() above

@app.get("/liteavatar/avatars")
async def liteavatar_avatars():
    """List available LiteAvatar avatars.

    Returns array of {id, name, preview_url} for each avatar directory found.
    """
    avatars_dir = Path(__file__).parent / "renderers" / "liteavatar" / "avatars"
    if not avatars_dir.exists():
        return {"avatars": []}
    avatars = []
    for d in sorted(avatars_dir.iterdir()):
        if d.is_dir() and (d / "neutral_pose.npy").exists():
            preview = f"/liteavatar/preview/{d.name}" if (d / "preview.png").exists() else None
            avatars.append({"id": d.name, "name": d.name, "preview_url": preview})
    return {"avatars": avatars}

@app.get("/liteavatar/preview/{avatar_id}")
async def liteavatar_preview(avatar_id: str):
    """Serve preview image for a LiteAvatar avatar."""
    preview_path = Path(__file__).parent / "renderers" / "liteavatar" / "avatars" / avatar_id / "preview.png"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(str(preview_path), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

@app.post("/liteavatar/render")
async def liteavatar_render(file: UploadFile = File(...), avatar_id: str = "woman1"):
    """Run LiteAvatar on uploaded audio, return video URL.

    POST multipart with 'file' (WAV) + 'avatar_id' (optional) →
    {"video_url": "/assets/liteavatar/output.mp4", "status": "ok"}

    Requires:
      - Model weights downloaded (run renderers/liteavatar/download_models.sh)
      - Avatar assets at renderers/liteavatar/avatars/<avatar_id>/
    """
    avatar_dir = Path(__file__).parent / "renderers" / "liteavatar" / "avatars" / avatar_id
    if not avatar_dir.exists():
        return {"status": "error", "error": f"Avatar not found: {avatar_id}"}

    ok = await _liteavatar_ensure_engine(str(avatar_dir), avatar_id)
    if not ok or _liteavatar_engine is None:
        return {"status": "error", "error": "LiteAvatar failed to load. Run download_models.sh first."}

    output_dir = Path(__file__).parent / "assets" / "liteavatar"
    try:
        audio_bytes = await file.read()
        loop = asyncio.get_event_loop()
        video = await loop.run_in_executor(
            None, _liteavatar_engine.process_bytes, audio_bytes, str(output_dir)
        )
        if video:
            return {
                "status": "ok",
                "video_url": f"/assets/liteavatar/{video.name}",
            }
        return {"status": "error", "error": "Rendering failed"}
    except Exception as e:
        logger.error(f"LiteAvatar render error: {e}")
        return {"status": "error", "error": str(e)}


# ============================================================================
# WebSocket 端點
# ============================================================================

# WebSocket 超時設定
WS_IDLE_TIMEOUT = 30      # 30 秒無訊息則斷開
WS_RECEIVE_TIMEOUT = 300    # 5 分鐘處理一個訊息

# WebSocket 速率限制
import time as time_module
_rate_limit_storage: dict[int, list[float]] = {}
_rate_limit_lock = asyncio.Lock()
WS_RATE_LIMIT = 30          # 每個 client 每 window
WS_RATE_WINDOW = 60         # 時間窗口（秒）


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """
    JARVIS 對話 WebSocket。

    用法（JavaScript）：
      const ws = new WebSocket("ws://localhost:8000/ws/chat");

      // 發送文字
      ws.send(JSON.stringify({type: "text", content: "你好"}));

      // 接收回應
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.response);
      };
    """
    await websocket.accept()
    pipeline = await get_pipeline()
    client_id = id(websocket)
    last_activity = asyncio.get_event_loop().time()
    logger.info(f"Client {client_id} 連線")

    # ===== 速率限制 =====
    async with _rate_limit_lock:
        now = time_module.time()
        if client_id not in _rate_limit_storage:
            _rate_limit_storage[client_id] = []
        # 清理過時記錄
        _rate_limit_storage[client_id] = [
            t for t in _rate_limit_storage[client_id]
            if now - t < WS_RATE_WINDOW
        ]
        if len(_rate_limit_storage[client_id]) >= WS_RATE_LIMIT:
            logger.warning(f"Client {client_id} 速率超限")
            await websocket.close(code=1008, reason="rate limit exceeded")
            return
        _rate_limit_storage[client_id].append(now)

    async def check_idle_timeout():
        """檢查是否超過 idle timeout"""
        nonlocal last_activity
        elapsed = asyncio.get_event_loop().time() - last_activity
        if elapsed > WS_IDLE_TIMEOUT:
            logger.warning(f"Client {client_id} idle timeout ({elapsed:.0f}s)")
            await websocket.close(code=1008, reason="idle timeout")
            return True
        return False

    try:
        while True:
            # 檢查 idle timeout
            if await check_idle_timeout():
                break

            # 設定 receive timeout
            try:
                raw = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=WS_RECEIVE_TIMEOUT
                )
                last_activity = asyncio.get_event_loop().time()
            except asyncio.TimeoutError:
                logger.warning(f"Client {client_id} receive timeout")
                await websocket.close(code=1008, reason="receive timeout")
                break
            except Exception:
                break

            # 速率限制檢查（每個訊息）
            async with _rate_limit_lock:
                now = time_module.time()
                _rate_limit_storage[client_id] = [
                    t for t in _rate_limit_storage[client_id]
                    if now - t < WS_RATE_WINDOW
                ]
                if len(_rate_limit_storage[client_id]) >= WS_RATE_LIMIT:
                    logger.warning(f"Client {client_id} 訊息速率超限")
                    await websocket.send_json({
                        "type": "error",
                        "error": "rate limit exceeded, please slow down",
                    })
                    await websocket.close(code=1008, reason="rate limit")
                    return
                _rate_limit_storage[client_id].append(now)

            # 處理 text frame
            if "text" in raw:
                await handle_text(websocket, pipeline, raw["text"])

            # 處理 binary frame（WAV 音頻）
            elif "bytes" in raw:
                await handle_voice(websocket, pipeline, raw["bytes"])

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} 斷線")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })
        except Exception as e:
            logger.warning(f"無法傳送錯誤訊息給 client: {e}")


async def handle_text(websocket: WebSocket, pipeline, raw_text: str):
    """處理文字訊息。"""
    try:
        msg = json.loads(raw_text)
    except json.JSONDecodeError:
        await websocket.send_json({
            "type": "error",
            "error": "Invalid JSON",
        })
        return

    msg_type = msg.get("type", "")
    if msg_type not in ("ping", "pong"):
        logger.info(f"收到文字訊息: {msg_type}")

    if msg_type == "text":
        content = msg.get("content", "").strip()
        if not content:
            await websocket.send_json({
                "type": "error",
                "error": "content is empty",
            })
            return

        current_model = os.environ.get("BRAIN_MODEL", "mnn-qwen1.5")
        cfgs = _load_models_config()["backends"]

        # 若 BRAIN_MODEL 是目錄路徑，轉回 config key
        if current_model not in cfgs:
            for k, v in cfgs.items():
                if v.get("model_dir", "").rstrip("/") == current_model.rstrip("/"):
                    current_model = k
                    break

        cfg = cfgs.get(current_model, {})
        if cfg.get("type") == "openai":
            result = await _openai_chat(content, current_model)
        elif cfg.get("agent") == "langchain":
            from brain.langchain_adapter import create_jarvis_agent, auto_scan_and_load_skills, MNNChatModel
            dir_map = {k: v.get("model_dir","") for k,v in cfgs.items()}
            mp = dir_map.get(current_model, "")
            if mp:
                mp = str(Path(__file__).parent / mp)
            llm = MNNChatModel(model_path=mp)
            tools = auto_scan_and_load_skills(str(Path(__file__).parent / "skills"))
            # 讀取 system prompt
            system_prompt = ""  # langchain 有自己的 SystemMessage
            config_path = Path.home() / ".jarvis_config.json"
            if config_path.exists():
                try:
                    with open(config_path, encoding="utf-8") as f:
                        root = json.load(f)
                    system_prompt = root.get("system_prompt", "").strip()
                except Exception:
                    pass
            agent = create_jarvis_agent(llm, tools, system_prompt=system_prompt)
            reply = agent(content)

            # TTS + viseme
            audio = None
            viseme_track = None
            try:
                from voice.voice_engine import get_engine
                tts = get_engine()
                arr, sr = tts.speak_to_array(reply)
                import io, wave, numpy as np

                # Generate viseme from audio
                try:
                    from render.render_engine import get_engine as get_render_engine
                    reng = get_render_engine(fps=30)
                    track = reng.render_from_audio(arr, sr)
                    viseme_track = reng.render_to_json(track)
                except Exception as ve:
                    logger.warning(f"Viseme 生成失敗 (non-fatal): {ve}")

                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                    w.writeframes((arr * 32767).astype(np.int16).tobytes())
                audio = buf.getvalue()
            except Exception as e:
                logger.warning(f"TTS 生成失敗: {e}")
            result = PipelineResult(transcription=content, response=reply, audio=audio, viseme_track=viseme_track)
        else:
            # MNN 本地模型（預設）
            if current_model in cfgs:
                os.environ["BRAIN_MODEL"] = cfgs[current_model].get("model_dir", "models_qwen1.5") + "/"
            result = await pipeline.run_text(content)
        await websocket.send_json(result.to_ws_message())

    elif msg_type == "ping":
        logger.debug("ping")
        await websocket.send_json({"type": "pong"})

    else:
        await websocket.send_json({
            "type": "error",
            "error": f"Unknown message type: {msg_type}",
        })


async def handle_voice(websocket: WebSocket, pipeline, wav_bytes: bytes):
    """處理語音訊息（WAV binary）。"""
    logger.info(f"收到語音資料: {len(wav_bytes)} bytes")

    if len(wav_bytes) < 1000:
        await websocket.send_json({
            "type": "error",
            "error": "Audio too short",
        })
        return

    # 安全性檢查：WAV 大小限制
    if len(wav_bytes) > MAX_WAV_SIZE:
        await websocket.send_json({
            "type": "error",
            "error": f"Audio too large (max {MAX_WAV_SIZE // 1024 // 1024}MB)",
        })
        return

    try:
        result = await pipeline.run_voice(wav_bytes)
        await websocket.send_json(result.to_ws_message())
    except Exception as e:
        logger.error(f"Voice pipeline error: {e}")
        await websocket.send_json({
            "type": "error",
            "error": str(e),
        })


# ============================================================================
# 啟動
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    cert_file = Path(__file__).parent / "server.crt"
    key_file = Path(__file__).parent / "server.key"
    ssl_kwargs = {}
    if cert_file.exists() and key_file.exists():
        ssl_kwargs = {
            "ssl_certfile": str(cert_file),
            "ssl_keyfile": str(key_file),
        }
        port = 8443
        print("🔒 HTTPS 已啟用 (port 8443)")
    else:
        port = 8000
        print("⚠️  使用 HTTP (port 8000)")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        **ssl_kwargs,
    )
