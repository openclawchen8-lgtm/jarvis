"""
LiteAvatar Engine — Audio-driven 2D avatar video renderer.

Wraps HumanAIGC/lite-avatar into the JARVIS pipeline.
Produces video frames from audio by:
  1. Paraformer feature extraction
  2. ONNX audio→mouth parameter prediction (32-d)
  3. TorchScript face generator → mouth region rendered onto background

Usage:
    engine = LiteAvatarEngine(avatar_dir="path/to/avatar")
    engine.load()
    video_path = engine.process(audio_wav_path, output_dir="results")
    # Returns path to .mp4
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("jarvis.liteavatar")

_SRC_DIR = Path(__file__).parent.parent / "liteavatar-src"


class LiteAvatarEngine:
    def __init__(
        self,
        avatar_dir: str,
        src_dir: str | None = None,
        num_threads: int = 2,
        fps: int = 30,
        use_gpu: bool = False,
    ):
        self.avatar_dir = Path(avatar_dir)
        self.src_dir = Path(src_dir) if src_dir else _SRC_DIR
        self.num_threads = num_threads
        self.fps = fps
        self.use_gpu = use_gpu
        self.device = "mps" if use_gpu else "cpu"
        self._avatar = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def validate_avatar(self) -> list[str]:
        """Check if avatar dir has all required files. Returns list of missing files."""
        required = [
            "neutral_pose.npy",
            "bg_video.mp4",
            "face_box.txt",
            "net_encode.pt",
            "net_decode.pt",
            "ref_frames",
        ]
        missing = []
        for f in required:
            p = self.avatar_dir / f
            if f == "ref_frames":
                if not p.is_dir():
                    missing.append(f)
            elif not p.exists():
                missing.append(f)
        return missing

    def load(self) -> bool:
        """Initialize LiteAvatar. Returns True on success."""
        try:
            missing = self.validate_avatar()
            if missing:
                logger.warning(f"LiteAvatar avatar missing files: {missing}")
                logger.warning("Download an avatar from ModelScope: "
                               "HumanAIGC-Engineering/LiteAvatarGallery")
                self._loaded = False
                return False

            sys.path.insert(0, str(self.src_dir))

            from lite_avatar import liteAvatar

            import torch
            if self.use_gpu and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            self._avatar = liteAvatar(
                data_dir=str(self.avatar_dir),
                language="ZH",
                num_threads=self.num_threads,
                fps=self.fps,
                use_bg_as_idle=False,
                generate_offline=True,
                use_gpu=(device == "mps"),
            )
            self._loaded = True
            logger.info(f"LiteAvatar loaded: {self.avatar_dir}")
            return True

        except Exception as e:
            logger.error(f"LiteAvatar load failed: {e}", exc_info=True)
            self._loaded = False
            return False

    def process(self, audio_path: str | Path, output_dir: str | Path) -> Optional[Path]:
        """Run LiteAvatar on audio file, produce MP4 video.

        Args:
            audio_path: Path to input WAV file (16kHz mono).
            output_dir: Directory for output video.

        Returns:
            Path to output .mp4 file, or None on failure.
        """
        if not self._loaded:
            logger.error("LiteAvatar not loaded")
            return None

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._avatar.handle(
                audio_file_path=str(audio_path),
                result_dir=str(output_dir),
            )
            video = output_dir / "test_demo.mp4"
            if video.exists():
                return video.resolve()
            logger.error(f"LiteAvatar output not found at {video}")
            return None
        except Exception as e:
            logger.error(f"LiteAvatar render failed: {e}", exc_info=True)
            return None

    def process_bytes(
        self, audio_bytes: bytes, output_dir: str | Path, filename: str = "input.wav"
    ) -> Optional[Path]:
        """Run LiteAvatar from WAV bytes."""
        import wave

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_path = output_dir / filename
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio_bytes)
        return self.process(wav_path, output_dir)

    def close(self):
        self._avatar = None
        self._loaded = False
