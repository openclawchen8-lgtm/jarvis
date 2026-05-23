"""
JARVIS Jitsi Audio — T021 (Capture) / T022 (Playback)

透過 Playwright 注入 JavaScript 到 Jitsi Meet 頁面，實現：
  - T022: TTS 音訊播放到會議（replaceTrack 虛擬麥克風）
  - T021: 遠端與會者音訊擷取 → ASR 管線
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("jarvis.jitsi.audio")

# ============================================================================
# JavaScript 片段
# ============================================================================

JS_FIND_AUDIO_SENDER = """
() => {
    const conf = window.APP && window.APP.conference;
    if (!conf || !conf._room) return null;
    const room = conf._room._room || conf._room;
    const pcs = [];
    if (room._controller && room._controller._peerConnection) {
        pcs.push(room._controller._peerConnection);
    }
    for (const pc of pcs) {
        const senders = pc.getSenders();
        const s = senders.find(s => s.track && s.track.kind === 'audio');
        if (s) return true;
    }
    return null;
}
"""

# T022: Play TTS audio into Jitsi meeting via replaceTrack
JS_PLAY_AUDIO = """
async (wavBase64) => {
    const conf = window.APP && window.APP.conference;
    if (!conf || !conf._room) return 'ERR_NO_CONF';
    const room = conf._room._room || conf._room;
    if (!room._controller || !room._controller._peerConnection) return 'ERR_NO_PC';
    const pc = room._controller._peerConnection;
    const senders = pc.getSenders();
    const audioSender = senders.find(s => s.track && s.track.kind === 'audio');
    if (!audioSender) return 'ERR_NO_AUDIO_SENDER';
    const oldTrack = audioSender.track;
    try {
        const ctx = new AudioContext();
        const resp = await fetch('data:audio/wav;base64,' + wavBase64);
        const buf = await resp.arrayBuffer();
        const audioBuf = await ctx.decodeAudioData(buf);
        const src = ctx.createBufferSource();
        src.buffer = audioBuf;
        const dst = ctx.createMediaStreamDestination();
        src.connect(dst);
        const newTrack = dst.stream.getAudioTracks()[0];
        await audioSender.replaceTrack(newTrack);
        logger.info('[JARVIS] TTS playback started');
        await new Promise((resolve, reject) => {
            src.onended = resolve;
            src.onerror = reject;
            src.start();
        });
        await audioSender.replaceTrack(oldTrack);
        logger.info('[JARVIS] TTS playback done, track restored');
        ctx.close();
        return 'OK';
    } catch (e) {
        logger.error('[JARVIS] TTS playback error: ' + e.message);
        try { await audioSender.replaceTrack(oldTrack); } catch (_) {}
        return 'ERR_' + e.message;
    }
}
"""

# T021: Hook into Jitsi remote track events to capture audio
JS_CAPTURE_SETUP = """
() => {
    if (window.__jarvis_capture_active) return;
    window.__jarvis_capture_active = true;
    const conf = window.APP && window.APP.conference;
    if (!conf || !conf._room) return 'ERR_NO_CONF';
    const room = conf._room._room || conf._room;
    room.on('remoteTrackAdded', (track) => {
        if (track.getType() !== 'audio') return;
        const ms = track.getOriginalStream();
        if (!ms) return;
        const mediaStream = ms;
        try {
            const recorder = new MediaRecorder(mediaStream, {
                mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                    ? 'audio/webm;codecs=opus'
                    : 'audio/webm'
            });
            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    e.data.arrayBuffer().then(buf => {
                        const bytes = new Uint8Array(buf);
                        if (window.__jarvis_on_audio_chunk) {
                            window.__jarvis_on_audio_chunk(Array.from(bytes));
                        }
                    });
                }
            };
            recorder.start(2000);
            track.on('disposed', () => {
                if (recorder.state !== 'inactive') recorder.stop();
            });
        } catch (e) {
            logger.warn('[JARVIS] MediaRecorder error: ' + e.message);
        }
    });
    return 'OK';
}
"""

JS_CHECK_REMOTE_TRACKS = """
() => {
    const conf = window.APP && window.APP.conference;
    if (!conf || !conf._room) return [];
    const room = conf._room._room || conf._room;
    const remoteTracks = room.getRemoteTracks ? room.getRemoteTracks() : [];
    return remoteTracks.filter(t => t.getType() === 'audio').map(t => t.getParticipantId());
}
"""


# ============================================================================
# T022: Audio Playback
# ============================================================================

class JitsiAudioPlayer:
    """Play TTS audio into Jitsi meeting via virtual microphone."""

    def __init__(self, page):
        self._page = page
        self._has_audio_sender = False

    def check_available(self) -> bool:
        """Check if Jitsi page has an audio RTCRtpSender we can replace."""
        try:
            result = self._page.evaluate(JS_FIND_AUDIO_SENDER)
            self._has_audio_sender = result is True
            return self._has_audio_sender
        except Exception as e:
            logger.warning(f"Audio sender check failed: {e}")
            return False

    def play_wav(self, wav_bytes: bytes) -> str:
        """Play WAV audio bytes into the Jitsi meeting.

        Returns 'OK' on success, or error string.
        """
        b64 = base64.b64encode(wav_bytes).decode()
        try:
            result = self._page.evaluate(JS_PLAY_AUDIO, b64)
            logger.info(f"TTS playback result: {result}")
            return result
        except Exception as e:
            logger.error(f"TTS playback failed: {e}")
            return f"ERR_PYTHON: {e}"

    def play_array(self, audio_arr: np.ndarray, sr: int) -> str:
        """Play numpy float32 audio array into Jitsi meeting.

        Args:
            audio_arr: float32 audio data in [-1, 1]
            sr: sample rate
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((audio_arr * 32767).astype(np.int16).tobytes())
        return self.play_wav(buf.getvalue())


# ============================================================================
# T021: Audio Capture
# ============================================================================

class JitsiAudioCapture:
    """Capture remote participants' audio from Jitsi meeting and feed to ASR."""

    def __init__(self, page, on_chunk: Optional[Callable[[bytes], None]] = None):
        self._page = page
        self._on_chunk = on_chunk
        self._active = False
        self._chunk_queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def start(self) -> str:
        """Inject capture JS into the Jitsi page.

        Remote audio chunks will be sent to the callback or internal queue.
        """
        if self._active:
            return "ALREADY_ACTIVE"

        # Expose Python callback to browser JS
        def on_chunk_from_js(byte_list: list):
            chunk = bytes(byte_list)
            if self._on_chunk:
                try:
                    self._on_chunk(chunk)
                except Exception as e:
                    logger.warning(f"Audio chunk callback error: {e}")
            try:
                self._chunk_queue.put_nowait(chunk)
            except Exception:
                pass

        self._page.expose_function("__jarvis_on_audio_chunk", on_chunk_from_js)

        try:
            result = self._page.evaluate(JS_CAPTURE_SETUP)
            self._active = result == "OK"
            if self._active:
                logger.info("Audio capture started ✅")
            else:
                logger.warning(f"Audio capture setup returned: {result}")
            return result
        except Exception as e:
            logger.error(f"Audio capture setup failed: {e}")
            self._active = False
            return f"ERR: {e}"

    def stop(self):
        """Stop audio capture."""
        if not self._active:
            return
        try:
            self._page.evaluate("window.__jarvis_capture_active = false;")
        except Exception:
            pass
        self._active = False
        logger.info("Audio capture stopped")

    @property
    def is_active(self) -> bool:
        return self._active

    async def get_chunk(self) -> bytes:
        """Async read next audio chunk from queue."""
        return await self._chunk_queue.get()

    def get_remote_participants(self) -> list:
        """List remote participant IDs that have audio tracks."""
        try:
            return self._page.evaluate(JS_CHECK_REMOTE_TRACKS) or []
        except Exception as e:
            logger.warning(f"Get remote tracks failed: {e}")
            return []


# ============================================================================
# Convenience: full audio pipeline for Jitsi bot
# ============================================================================

class JitsiAudioBridge:
    """High-level bridge combining capture + playback + ASR/LLM/TTS pipeline.

    Usage:
        bridge = JitsiAudioBridge(page)
        bridge.set_pipeline(pipeline)
        await bridge.start_capture()
        # ... main loop ...
        bridge.say("Hello everyone")  # TTS → Jitsi
    """

    def __init__(self, page):
        self.page = page
        self.player = JitsiAudioPlayer(page)
        self.capture = JitsiAudioCapture(page)
        self._pipeline = None
        self._pending_responses: list[dict] = []
        self._capture_task: Optional[asyncio.Task] = None

    def set_pipeline(self, pipeline):
        """Set JarvisPipeline instance for ASR on captured audio."""
        self._pipeline = pipeline

    def say(self, text: str) -> str:
        """Speak text into Jitsi meeting (TTS → audio → replaceTrack).

        Uses the pipeline's TTS to generate audio, then plays into Jitsi.
        """
        if self._pipeline is None:
            return "ERR_NO_PIPELINE"

        # Generate TTS audio
        from voice.voice_engine import get_engine
        engine = get_engine()
        arr, sr = engine.speak_to_array(text)
        return self.player.play_array(arr, sr)

    def say_array(self, audio_arr: np.ndarray, sr: int) -> str:
        """Play pre-generated audio array into Jitsi."""
        return self.player.play_array(audio_arr, sr)

    def say_wav(self, wav_bytes: bytes) -> str:
        """Play pre-generated WAV bytes into Jitsi."""
        return self.player.play_wav(wav_bytes)

    async def start_capture(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start capturing remote audio and feeding it through ASR pipeline."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not set. Call set_pipeline() first.")

        result = await self.capture.start()
        if result != "OK":
            raise RuntimeError(f"Capture start failed: {result}")

        # Background task: process incoming audio chunks through ASR
        loop = loop or asyncio.get_event_loop()
        self._capture_task = loop.create_task(self._capture_loop())
        logger.info("Capture processing loop started")

    async def _capture_loop(self):
        """Background loop: collect audio chunks, run ASR when enough data."""
        buffer = bytearray()
        min_chunk_duration = 1.0  # seconds
        sample_rate = 48000  # webm/opus typical rate

        while self.capture.is_active:
            try:
                chunk = await asyncio.wait_for(
                    self.capture.get_chunk(), timeout=1.0
                )
                buffer.extend(chunk)

                # Simple heuristic: process when buffer is large enough
                # webm ~20KB/s for opus at 48kHz
                if len(buffer) > 40000:  # ~2 seconds
                    await self._process_buffer(bytes(buffer))
                    buffer.clear()
            except asyncio.TimeoutError:
                # Check if we have accumulated enough
                if len(buffer) > 20000:
                    await self._process_buffer(bytes(buffer))
                    buffer.clear()
                continue
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                break

    async def _process_buffer(self, audio_chunk: bytes):
        """Run ASR on captured audio chunk and queue response."""
        if self._pipeline is None:
            return
        try:
            # Try to decode webm to PCM WAV
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f_in:
                f_in.write(audio_chunk)
                webm_path = f_in.name

            wav_path = webm_path + ".wav"
            try:
                # Use ffmpeg to convert webm → wav
                subprocess.run(
                    ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000",
                     "-ac", "1", "-sample_fmt", "s16", wav_path],
                    capture_output=True, timeout=30,
                )
                with open(wav_path, "rb") as f:
                    wav_bytes = f.read()
                if len(wav_bytes) > 2000:
                    result = await self._pipeline.run_voice(wav_bytes)
                    if result.response:
                        # Queue response — bridge main loop will handle playback
                        self._pending_responses.append({
                            "transcription": result.transcription,
                            "response": result.response,
                            "audio": result.audio,
                        })
                        logger.info(f"ASR: {result.transcription} → {result.response[:60]}")
            finally:
                for p in [webm_path, wav_path]:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Buffer processing error: {e}")

    def stop_capture(self):
        """Stop capture and cleanup."""
        self.capture.stop()
        if self._capture_task:
            self._capture_task.cancel()
            self._capture_task = None

    def get_pending_responses(self) -> list[dict]:
        """Get and clear pending ASR responses."""
        responses = self._pending_responses[:]
        self._pending_responses.clear()
        return responses

    def has_pending(self) -> bool:
        return len(self._pending_responses) > 0
