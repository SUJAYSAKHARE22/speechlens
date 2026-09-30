"""Low-level audio helpers: I/O, resampling, levels, fades and hashing."""

from __future__ import annotations

import hashlib
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

EPS = 1e-12


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load any libsndfile-readable file as mono float64 at ``sample_rate``."""
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr != sample_rate:
        g = gcd(int(sr), int(sample_rate))
        x = resample_poly(x, sample_rate // g, int(sr) // g)
    return np.ascontiguousarray(x)


def to_pcm16(x: np.ndarray) -> np.ndarray:
    """Quantise float audio to int16 (the exact representation we store and hash)."""
    return np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")


def pcm16_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(to_pcm16(x).tobytes()).hexdigest()


def save_audio(path: str | Path, x: np.ndarray, sample_rate: int) -> None:
    """Write mono 16-bit audio; format is inferred from the extension (wav/flac)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), to_pcm16(x), sample_rate, subtype="PCM_16")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + EPS)) if len(x) else 0.0


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(rms(x) + EPS)


def db_to_gain(db: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(db) / 20.0)


def fade_edges(x: np.ndarray, n: int) -> np.ndarray:
    """Return a copy of ``x`` with raised-cosine fades of ``n`` samples on both ends."""
    y = x.copy()
    n = min(n, len(y) // 2)
    if n > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def ms_to_samples(ms: float, sample_rate: int) -> int:
    return int(round(ms * sample_rate / 1000.0))
