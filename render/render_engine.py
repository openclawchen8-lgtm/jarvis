"""
Viseme-based Digital Human Render Engine.

輕量級口型驅動引擎：將 phoneme/音頻映射為 viseme 動畫幀，
可在 Mac M2 上即時執行（無 GPU 需求）。

Improvements over v1:
  - Smooth viseme transitions with easing
  - Co-articulation smoothing (look-ahead blending)
  - Blink scheduling at silence points + periodic
  - Breathing-inspired intensity modulation
  - Hysteresis-based energy state machine (no random jumps)

用法：
    engine = RenderEngine()
    frames = engine.render_from_phonemes(phoneme_seq, audio_duration)
    frames = engine.render_from_audio(audio_array, sample_rate)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("jarvis.render")

VISEME_COUNT = 12


class Viseme(IntEnum):
    """基本嘴型（對應 IPA phoneme 分類）。"""
    REST   = 0   # 閉合/靜止
    A      = 1   # aa, ae, ah 張大
    B      = 2   # b, p, m 閉合
    C      = 3   # ch, jh, sh 噘嘴
    D      = 4   # d, t, n 微張
    E      = 5   # eh, er 扁平
    F      = 6   # f, v 下唇上齒
    G      = 7   # g, k, h 張開
    I      = 8   # iy, ih 微笑
    O      = 9   # ow, ao 圓形
    U      = 10  # uw, uh 小圓
    W      = 11  # w 噘圓


# Phoneme → Viseme 映射表
PHONEME_TO_VISEME: dict[str, Viseme] = {
    "aa": Viseme.A, "ae": Viseme.A, "ah": Viseme.A,
    "b": Viseme.B, "p": Viseme.B, "m": Viseme.B,
    "ch": Viseme.C, "jh": Viseme.C, "sh": Viseme.C,
    "d": Viseme.D, "t": Viseme.D, "n": Viseme.D, "dx": Viseme.D,
    "eh": Viseme.E, "er": Viseme.E, "en": Viseme.E,
    "f": Viseme.F, "v": Viseme.F,
    "g": Viseme.G, "k": Viseme.G, "h": Viseme.G, "hh": Viseme.G, "ng": Viseme.G,
    "iy": Viseme.I, "ih": Viseme.I, "ix": Viseme.I,
    "ow": Viseme.O, "ao": Viseme.O, "oy": Viseme.O,
    "uw": Viseme.U, "uh": Viseme.U,
    "w": Viseme.W,
    "s": Viseme.C, "z": Viseme.C, "zh": Viseme.C,
    "th": Viseme.D, "dh": Viseme.D,
    "l": Viseme.D, "r": Viseme.D,
    "y": Viseme.I,
    "": Viseme.REST, "sil": Viseme.REST, "sp": Viseme.REST,
}

# Viseme openness level (0.0 = closed, 1.0 = wide open)
VISEME_OPENNESS: dict[Viseme, float] = {
    Viseme.REST: 0.0,
    Viseme.B: 0.0,
    Viseme.F: 0.15,
    Viseme.D: 0.25,
    Viseme.E: 0.2,
    Viseme.U: 0.3,
    Viseme.W: 0.35,
    Viseme.C: 0.4,
    Viseme.O: 0.5,
    Viseme.I: 0.35,
    Viseme.G: 0.7,
    Viseme.A: 0.85,
}


@dataclass
class VisemeFrame:
    """單幀 viseme 動畫資料（所有數值存 Python 原生型別）。"""
    viseme_id: int
    timestamp: float
    intensity: float = 1.0
    openness: float = 0.0
    roundness: float = 0.0

    def __post_init__(self):
        self.viseme_id = int(self.viseme_id)
        self.timestamp = float(self.timestamp)
        self.intensity = float(self.intensity)
        self.openness = float(self.openness)
        self.roundness = float(self.roundness)


@dataclass
class VisemeTrack:
    """完整的 viseme 時間軸。"""
    frames: List[VisemeFrame] = field(default_factory=list)
    duration: float = 0.0
    fps: int = 30
    blink_timestamps: List[float] = field(default_factory=list)
    has_audio: bool = False


BLINK_MIN_INTERVAL = 2.5
BLINK_MAX_INTERVAL = 5.5
BLINK_DURATION = 0.12
BREATH_RATE = 0.25
BREATH_AMPLITUDE = 0.06
CO_ARTICULATION_WINDOW = 3
SMOOTHING_FRAMES = 4


class RenderEngine:
    """
    Viseme-based 口型渲染引擎（v2）。

    使用方式：
        engine = RenderEngine(fps=30)
        track = engine.render_from_phonemes(["aa", "b", "sil"], duration=2.0)
        track = engine.render_from_audio(audio_array, sample_rate)
    """

    def __init__(self, fps: int = 30):
        self.fps = fps
        self._loaded = False

    def load(self) -> bool:
        self._loaded = True
        logger.info("RenderEngine v2 已就緒（viseme-based）")
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def close(self):
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_from_phonemes(
        self,
        phonemes: List[str],
        duration: float,
    ) -> VisemeTrack:
        """
        從 phoneme 序列產生 viseme 時間軸（含平滑與自然化）。
        """
        if not phonemes:
            return VisemeTrack(frames=[], duration=duration, fps=self.fps)

        dt = 1.0 / self.fps
        n_frames = max(int(duration * self.fps), len(phonemes))

        coarse = self._build_coarse_frames(phonemes, n_frames, dt)
        smoothed = self._apply_co_articulation(coarse)
        blended = self._smooth_transitions(smoothed, dt)
        track = self._add_breathing_and_blinks(blended, dt, has_audio=False)
        track.duration = duration
        track.fps = self.fps
        return track

    def render_from_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> VisemeTrack:
        """
        從音頻能量產生 viseme 時間軸。

        使用帶遲滯的狀態機來平滑分配 viseme，避免隨機跳躍。
        """
        duration = len(audio) / sample_rate
        dt = 1.0 / self.fps
        n_frames = int(duration * self.fps)

        coarse = self._energy_to_viseme_frames(audio, sample_rate, n_frames, dt)
        smoothed = self._apply_co_articulation(coarse)
        blended = self._smooth_transitions(smoothed, dt)
        track = self._add_breathing_and_blinks(blended, dt, audio=audio, sample_rate=sample_rate)
        track.duration = duration
        track.fps = self.fps
        track.has_audio = True
        return track

    def render_to_json(self, track: VisemeTrack) -> dict:
        """序列化為 JSON 供前端消費（全部轉 Python 原生型別）。"""
        return {
            "type": "viseme_track",
            "fps": track.fps,
            "duration": float(round(track.duration, 2)),
            "has_audio": track.has_audio,
            "blinks": [float(round(t, 3)) for t in track.blink_timestamps],
            "frames": [
                {
                    "v": int(f.viseme_id),
                    "t": float(round(f.timestamp, 3)),
                    "i": float(round(f.intensity, 2)),
                    "o": float(round(f.openness, 2)),
                    "r": float(round(f.roundness, 2)),
                }
                for f in track.frames
            ],
        }

    def generate_viseme_ids(self, track: VisemeTrack) -> List[int]:
        return [f.viseme_id for f in track.frames]

    def generate_timestamps(self, track: VisemeTrack) -> List[float]:
        return [f.timestamp for f in track.frames]

    # ------------------------------------------------------------------
    # Internal: build coarse frames from phonemes
    # ------------------------------------------------------------------

    def _build_coarse_frames(
        self, phonemes: List[str], n_frames: int, dt: float
    ) -> List[VisemeFrame]:
        """Distribute phonemes evenly across frames."""
        frames_per_phoneme = max(n_frames // len(phonemes), 1)
        frames: List[VisemeFrame] = []
        for i, ph in enumerate(phonemes):
            viseme = PHONEME_TO_VISEME.get(ph.lower(), Viseme.REST)
            t_start = i * frames_per_phoneme * dt
            for j in range(frames_per_phoneme):
                t = t_start + j * dt
                openness = VISEME_OPENNESS.get(viseme, 0.0)
                frames.append(VisemeFrame(
                    viseme_id=int(viseme),
                    timestamp=round(t, 3),
                    intensity=1.0,
                    openness=openness,
                    roundness=0.5 if viseme in (Viseme.O, Viseme.U, Viseme.W) else 0.0,
                ))
        # fill remaining with REST
        if frames:
            last_t = frames[-1].timestamp
            while last_t < n_frames * dt:
                last_t += dt
                frames.append(VisemeFrame(
                    viseme_id=int(Viseme.REST),
                    timestamp=round(last_t, 3),
                    intensity=0.0,
                    openness=0.0,
                    roundness=0.0,
                ))
        return frames

    # ------------------------------------------------------------------
    # Internal: energy-based viseme state machine (hysteresis)
    # ------------------------------------------------------------------

    def _energy_to_viseme_frames(
        self,
        audio: np.ndarray,
        sample_rate: int,
        n_frames: int,
        dt: float,
    ) -> List[VisemeFrame]:
        """
        Map audio energy to viseme frames using a state machine with hysteresis.
        
        States: SILENT, CLOSED, MID, OPEN, WIDE
        Transitions have hysteresis to prevent rapid flickering.
        """
        frame_len = int(sample_rate / self.fps)
        frames: List[VisemeFrame] = []

        # Pre-compute per-frame RMS energy
        energies = []
        for i in range(n_frames):
            start = i * frame_len
            end = min(start + frame_len, len(audio))
            seg = audio[start:end]
            if len(seg) == 0:
                rms = 0.0
            else:
                rms = np.sqrt(np.mean(seg ** 2))
            energies.append(rms)

        if not energies:
            return frames

        # Normalize energy to [0, 1]
        max_e = max(energies) if max(energies) > 0 else 1.0
        norm_energies = [min(e / max_e, 1.0) for e in energies]

        # Thresholds with hysteresis
        TH_LOW_ON = 0.06
        TH_LOW_OFF = 0.03
        TH_MID_ON = 0.20
        TH_MID_OFF = 0.12
        TH_HIGH_ON = 0.45
        TH_HIGH_OFF = 0.30

        # Smooth energy with exponential moving average
        smooth_e = 0.0
        alpha = 0.3

        # Viseme candidates per energy level
        # (viseme, openness, roundness)
        closed_visemes = [(Viseme.REST, 0.0, 0.0), (Viseme.B, 0.0, 0.0)]
        closed_visemes += [(Viseme.F, 0.15, 0.0)]
        closed_visemes += [(Viseme.D, 0.25, 0.0)]

        mid_visemes = [(Viseme.U, 0.3, 0.6), (Viseme.I, 0.35, 0.0)]
        mid_visemes += [(Viseme.E, 0.2, 0.0), (Viseme.C, 0.4, 0.5)]

        open_visemes = [(Viseme.O, 0.5, 0.8), (Viseme.G, 0.7, 0.0)]

        wide_visemes = [(Viseme.A, 0.85, 0.0)]

        state = "silent"
        hold_counter = 0
        HOLD_FRAMES = 3

        for i in range(n_frames):
            smooth_e = alpha * norm_energies[i] + (1 - alpha) * smooth_e
            t = i * dt

            # State machine with hysteresis
            if state == "silent":
                if smooth_e > TH_LOW_ON:
                    state = "closed"
                    hold_counter = 0
            elif state == "closed":
                if smooth_e > TH_MID_ON:
                    state = "mid"
                    hold_counter = 0
                elif smooth_e < TH_LOW_OFF:
                    hold_counter += 1
                    if hold_counter > HOLD_FRAMES:
                        state = "silent"
            elif state == "mid":
                if smooth_e > TH_HIGH_ON:
                    state = "open"
                    hold_counter = 0
                elif smooth_e < TH_MID_OFF:
                    hold_counter += 1
                    if hold_counter > HOLD_FRAMES:
                        state = "closed"
            elif state == "open":
                if smooth_e > 0.65:
                    state = "wide"
                    hold_counter = 0
                elif smooth_e < TH_HIGH_OFF:
                    hold_counter += 1
                    if hold_counter > HOLD_FRAMES:
                        state = "mid"
            elif state == "wide":
                if smooth_e < TH_HIGH_OFF:
                    hold_counter += 1
                    if hold_counter > HOLD_FRAMES:
                        state = "open"
            else:
                state = "silent"

            # Select viseme based on state + energy + randomness for variety
            intensity = min(1.0, smooth_e * 3.0)

            if state == "silent":
                v_id, openness, roundness = Viseme.REST, 0.0, 0.0
                intensity = 0.0
            elif state == "closed":
                idx = int(smooth_e * len(closed_visemes)) % len(closed_visemes)
                v_id, openness, roundness = closed_visemes[idx]
                intensity = max(0.1, intensity)
            elif state == "mid":
                idx = int(smooth_e * len(mid_visemes)) % len(mid_visemes)
                v_id, openness, roundness = mid_visemes[idx]
            elif state == "open":
                idx = int(smooth_e * len(open_visemes)) % len(open_visemes)
                v_id, openness, roundness = open_visemes[idx]
            elif state == "wide":
                idx = int(smooth_e * len(wide_visemes)) % len(wide_visemes)
                v_id, openness, roundness = wide_visemes[idx]
                intensity = 1.0
            else:
                v_id, openness, roundness = Viseme.REST, 0.0, 0.0
                intensity = 0.0

            frames.append(VisemeFrame(
                viseme_id=int(v_id),
                timestamp=round(t, 3),
                intensity=float(round(intensity, 2)),
                openness=float(round(openness, 2)),
                roundness=float(round(roundness, 2)),
            ))

        return frames

    # ------------------------------------------------------------------
    # Internal: co-articulation smoothing
    # ------------------------------------------------------------------

    def _apply_co_articulation(
        self, frames: List[VisemeFrame]
    ) -> List[VisemeFrame]:
        """
        Apply look-ahead co-articulation: blend each frame with future frames
        to anticipate upcoming visemes. Makes speech look more natural.
        """
        if len(frames) < 3:
            return frames

        smoothed = []
        half_win = CO_ARTICULATION_WINDOW // 2

        for i in range(len(frames)):
            # Gather openness/roundness from neighbors
            start = max(0, i - half_win)
            end = min(len(frames), i + half_win + 1)
            neighborhood = frames[start:end]

            openness = sum(f.openness for f in neighborhood) / len(neighborhood)
            roundness = sum(f.roundness for f in neighborhood) / len(neighborhood)

            f = frames[i]
            smoothed.append(VisemeFrame(
                viseme_id=f.viseme_id,
                timestamp=f.timestamp,
                intensity=f.intensity,
                openness=round(openness, 2),
                roundness=round(roundness, 2),
            ))

        return smoothed

    def _smooth_transitions(
        self, frames: List[VisemeFrame], dt: float
    ) -> List[VisemeFrame]:
        """
        Insert eased interpolation frames when viseme ID changes.
        Uses ease-in-out sine easing.
        """
        if len(frames) < 2:
            return frames

        result: List[VisemeFrame] = []
        for i in range(len(frames) - 1):
            curr = frames[i]
            nxt = frames[i + 1]
            result.append(curr)

            if curr.viseme_id != nxt.viseme_id:
                # Insert interpolation frames
                for j in range(1, SMOOTHING_FRAMES + 1):
                    t_factor = j / (SMOOTHING_FRAMES + 1)
                    # Ease-in-out sine
                    eased = 0.5 - 0.5 * math.cos(math.pi * t_factor)

                    interp_t = curr.timestamp + t_factor * (nxt.timestamp - curr.timestamp)
                    interp_v = curr.viseme_id if eased < 0.5 else nxt.viseme_id
                    interp_i = curr.intensity * (1 - eased) + nxt.intensity * eased
                    interp_o = curr.openness * (1 - eased) + nxt.openness * eased
                    interp_r = curr.roundness * (1 - eased) + nxt.roundness * eased

                    result.append(VisemeFrame(
                        viseme_id=interp_v,
                        timestamp=round(interp_t, 3),
                        intensity=round(interp_i, 2),
                        openness=round(interp_o, 2),
                        roundness=round(interp_r, 2),
                    ))

        result.append(frames[-1])
        return result

    # ------------------------------------------------------------------
    # Internal: breathing + blinks
    # ------------------------------------------------------------------

    def _add_breathing_and_blinks(
        self,
        frames: List[VisemeFrame],
        dt: float,
        audio: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
        has_audio: bool = False,
    ) -> VisemeTrack:
        """
        Modulate intensity with breathing pattern and schedule blinks.
        Blinks at silence points if audio provided, otherwise periodic.
        """
        if not frames:
            return VisemeTrack(frames=[], duration=0.0, fps=self.fps)

        # Breathing modulation
        for i, f in enumerate(frames):
            t = f.timestamp
            breath = 1.0 + BREATH_AMPLITUDE * math.sin(2 * math.pi * BREATH_RATE * t)
            f.intensity = round(min(1.0, f.intensity * breath), 2)

        # Blink scheduling
        blinks = self._schedule_blinks(frames, dt, audio, sample_rate)

        return VisemeTrack(
            frames=frames,
            duration=frames[-1].timestamp if frames else 0.0,
            fps=self.fps,
            blink_timestamps=blinks,
            has_audio=has_audio,
        )

    def _schedule_blinks(
        self,
        frames: List[VisemeFrame],
        dt: float,
        audio: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
    ) -> List[float]:
        """
        Generate natural-looking blink timestamps.
        Blinks happen:
          1. At silence/rest points in audio (preferred)
          2. Periodically every 2.5-5.5 seconds if no silence
        
        Returns list of timestamps (seconds).
        """
        blinks: List[float] = []
        if not frames:
            return blinks

        # Find REST frames (candidate blink points)
        rest_times = [
            f.timestamp for f in frames
            if f.viseme_id == int(Viseme.REST) and f.intensity < 0.1
        ]

        # Also find silence from audio
        silence_times: List[float] = []
        if audio is not None and sample_rate is not None and len(audio) > 0:
            frame_len = int(sample_rate / self.fps)
            for i in range(len(frames)):
                start = i * frame_len
                end = min(start + frame_len, len(audio))
                seg = audio[start:end]
                if len(seg) > 0:
                    rms = np.sqrt(np.mean(seg ** 2))
                    if rms < 0.015:
                        silence_times.append(frames[i].timestamp)

        # Merge candidates
        candidates = sorted(set(rest_times + silence_times))

        last_blink = -BLINK_MIN_INTERVAL
        for t in candidates:
            if t - last_blink >= BLINK_MIN_INTERVAL and t >= 0.3:
                blinks.append(round(t, 3))
                last_blink = t

        # Ensure periodic blinking even without good candidates
        total_duration = frames[-1].timestamp if frames else 0.0
        if total_duration > 1.0:
            t = BLINK_MIN_INTERVAL
            while t < total_duration - 0.3:
                # Check if we already have a blink near this time
                if not any(abs(b - t) < 0.5 for b in blinks):
                    # Find nearest candidate
                    nearby = [c for c in candidates if abs(c - t) < 0.5]
                    if nearby:
                        blinks.append(round(nearby[0], 3))
                    elif random.random() < 0.7:
                        blinks.append(round(t, 3))
                t += random.uniform(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL)

        return sorted(set(blinks))

    def _add_blink_frames(
        self, frames: List[VisemeFrame], blink_ts: List[float], dt: float
    ) -> List[VisemeFrame]:
        """Insert a brief REST frame at each blink timestamp (eye closure indicator)."""
        if not blink_ts or not frames:
            return frames

        result = list(frames)
        blink_half = int((BLINK_DURATION / 2) / dt)

        for bt in blink_ts:
            # Find nearest frame index
            idx = min(range(len(frames)), key=lambda i: abs(frames[i].timestamp - bt))
            # Insert blink: set nearby frames to REST with intensity 0
            for offset in range(-blink_half, blink_half + 1):
                pos = idx + offset
                if 0 <= pos < len(result):
                    result[pos].viseme_id = int(Viseme.REST)
                    result[pos].intensity = 0.0

        return result


_engine: Optional[RenderEngine] = None


def get_engine(fps: int = 30) -> RenderEngine:
    """取得全域 RenderEngine v2 實例。"""
    global _engine
    if _engine is None:
        _engine = RenderEngine(fps=fps)
    return _engine
