"""Speaker-adaptive F0 estimation (Praat via Parselmouth).

Uses the two-pass strategy of Hirst & De Looze: a wide first pass estimates the
speaker's range, then the analysis is repeated with floor = 0.75*Q1 and
ceiling = 1.5*Q3. This sharply reduces octave errors compared to fixed limits and
is what makes the downstream comparison speaker-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth


@dataclass(frozen=True)
class F0Range:
    floor: float
    ceiling: float
    median: float


def f0_contour(
    x: np.ndarray,
    sr: int,
    floor: float = 75.0,
    ceiling: float = 600.0,
    time_step: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, f0_hz)``; unvoiced frames are NaN."""
    snd = parselmouth.Sound(np.ascontiguousarray(x), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=time_step, pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 <= 0] = np.nan
    return pitch.xs(), f0


def estimate_f0_range(x: np.ndarray, sr: int) -> F0Range:
    """Two-pass speaker F0 range; falls back to a generic range for unvoiced input."""
    _, f0 = f0_contour(x, sr, 75.0, 600.0)
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < 20:
        return F0Range(75.0, 500.0, 150.0)
    q1, q3 = np.percentile(voiced, [25, 75])
    floor, ceiling = max(50.0, 0.75 * q1), min(800.0, 1.5 * q3)
    _, f0 = f0_contour(x, sr, floor, ceiling)
    voiced = f0[~np.isnan(f0)]
    median = float(np.median(voiced)) if len(voiced) else float((floor + ceiling) / 2)
    return F0Range(float(floor), float(ceiling), median)
