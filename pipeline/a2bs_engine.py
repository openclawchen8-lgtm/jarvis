"""
A2BS Engine — Audio-to-BlendShape using UniTalker-MNN (MNN + Metal).

Port of alibaba/MNN/apps/Android/MnnTaoAvatar a2bs C++ module to Python.
Returns FLAME blendshape coefficients: [expr_50, jaw_pose_3] per frame at 20fps.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("jarvis.a2bs")

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models_a2bs", "MNN", "UniTalker-MNN")


class A2BSEngine:
    # FIXME: Metal backend loads but produces all-zero output.
    # Some model ops lack Metal support. Use CPU for now.
    def __init__(self, model_dir: str = _MODEL_DIR, backend: str = "CPU"):
        self.model_dir = model_dir
        self.backend = backend
        self._interpreter = None
        self._session = None
        self._loaded = False

    def load(self) -> bool:
        model_path = os.path.join(self.model_dir, "audio2verts.mnn")
        if not os.path.exists(model_path):
            logger.error(f"A2BS model not found: {model_path}")
            return False

        import MNN
        self._interpreter = MNN.Interpreter(model_path)
        config = {"backend": self.backend, "numThread": 1}
        if self.backend == "METAL":
            config["precision"] = 2  # Precision_High
            config["memory"] = 1     # Memory_Low
            logger.info("A2BS using Metal backend (GPU)")
        else:
            logger.info("A2BS using CPU backend")
        self._session = self._interpreter.createSession(config)
        self._loaded = True
        logger.info(f"A2BS engine loaded: {model_path}")
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def close(self):
        if self._interpreter is not None:
            self._interpreter = None
        self._session = None
        self._loaded = False

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        ori_fps: int = 25,
        out_fps: int = 20,
        num_exp: int = 50,
    ) -> Optional[dict]:
        """
        Run A2BS on audio, return blendshape coefficients.

        Args:
            audio: float32 array in [-1.0, 1.0]
            sample_rate: input audio sample rate
            ori_fps: model's native fps (25)
            out_fps: output fps (20)

        Returns:
            dict with:
              - "coeffs": list of [expr_50 + jaw_pose_3] per frame
              - "fps": output fps
              - "num_frames": int
            or None on failure.
        """
        if not self._loaded:
            logger.warning("A2BS not loaded")
            return None
        if len(audio) == 0:
            logger.warning("Empty audio")
            return None

        # Step 1: Resample to 16kHz (same as C++ reference)
        audio_16k = _resample_audio(audio, sample_rate, 16000)

        # Step 2: Normalize (z-score)
        audio_norm = _normalize_audio(audio_16k)

        # Step 3: Inference
        coeffs = self._infer(audio_norm)
        if coeffs is None or len(coeffs) == 0:
            return None

        num_coeff = num_exp + 3  # 50 expr + 3 jaw_pose

        # If do_verts2flame is False, pad extra frame (matching C++ ref)
        # Here audio2verts.mnn predicts 53 dim directly (verts mode)
        # The output is [N, 53] where 53 = 50 expr + 3 jaw_pose
        if coeffs.shape[0] < 2:
            return None

        flat = coeffs.reshape(-1)

        # Step 4: Resample fps from ori_fps to out_fps
        n_rows = len(flat) // num_coeff
        coeffs_2d = flat[:n_rows * num_coeff].reshape(n_rows, num_coeff)

        n_out = int(n_rows / ori_fps * out_fps)
        resampled = _resample_bs_params(coeffs_2d, n_out)

        return {
            "coeffs": resampled.tolist(),
            "fps": out_fps,
            "num_frames": len(resampled),
        }

    def _infer(self, audio: np.ndarray) -> Optional[np.ndarray]:
        import MNN
        audio_len = len(audio)
        if audio_len == 0:
            return None

        input_tensor = self._interpreter.getSessionInput(self._session)
        self._interpreter.resizeTensor(input_tensor, (1, audio_len))
        self._interpreter.resizeSession(self._session)

        inp = audio.astype(np.float32).reshape(1, -1)
        input_tensor.fromNumpy(inp)

        self._interpreter.runSession(self._session)

        out = self._interpreter.getSessionOutput(self._session)
        data = out.getNumpyData()
        return data


# ============================================================================
# Audio utilities (ported from C++ a2bs_utils.cpp)
# ============================================================================

def _resample_audio(
    audio: np.ndarray, src_sr: int, dst_sr: int
) -> np.ndarray:
    """Resample audio to target sample rate using linear interpolation."""
    if src_sr == dst_sr:
        return audio
    duration = len(audio) / src_sr
    n_out = int(duration * dst_sr)
    x_old = np.linspace(0, len(audio) - 1, len(audio))
    x_new = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    mean = np.mean(audio)
    std = np.std(audio) + 1e-7
    return (audio - mean) / std


def _resample_bs_params(
    bs_params: np.ndarray, n_out: int
) -> np.ndarray:
    """Resample blendshape params from current fps to target fps."""
    n_in = bs_params.shape[0]
    n_dim = bs_params.shape[1]
    if n_in < 2:
        return bs_params
    x_old = np.linspace(0, n_in - 1, n_in)
    x_new = np.linspace(0, n_in - 1, n_out)
    out = np.zeros((n_out, n_dim), dtype=np.float32)
    for d in range(n_dim):
        out[:, d] = np.interp(x_new, x_old, bs_params[:, d])
    return out
