"""Calibrate the natural between-speaker tolerance from good-vs-good recordings.

Given deviation tables of *good* readings compared with the reference (different speakers,
same text), the natural spread of every metric is measured; a metric's sigma is set so that
its 95th-percentile natural deviation corresponds to |z| = 1.5. Calibration can only *widen*
the default tolerance, so a tiny calibration set can never make the detector over-sensitive.
"""

from __future__ import annotations

import numpy as np

from speechlens.analysis.compare import Deviations
from speechlens.config import Tolerance

NATURAL_P95_Z = 1.5


def _p95(values: list[np.ndarray], mask: list[np.ndarray] | None = None) -> float:
    v = np.concatenate([a[m] if mask else a for a, m in zip(values, mask or values, strict=True)])
    v = np.abs(v[np.isfinite(v) & (v != 0)])
    return float(np.percentile(v, 95)) if len(v) else 0.0


def calibrate_tolerance(good: list[Deviations], base: Tolerance | None = None) -> Tolerance:
    """Return a tolerance at least as wide as ``base`` and wide enough for the good readings."""
    base = base or Tolerance()
    if not good:
        return base
    pause_dev = [d.pause_obs - d.pause_exp for d in good]
    scale = {
        "pace_sigma": _p95([d.pace for d in good]) / NATURAL_P95_Z,
        "pitch_sigma": _p95([d.pitch for d in good]) / NATURAL_P95_Z,
        "terminal_sigma": _p95([d.terminal for d in good]) / NATURAL_P95_Z,
        "energy_sigma": _p95([d.energy for d in good]) / NATURAL_P95_Z,
        "clarity_sigma": _p95([d.clarity for d in good]) / NATURAL_P95_Z,
        # pauses: widen the absolute term only (the relative term already scales with the pause)
        "pause_sigma_abs": max(
            (_p95(pause_dev) / NATURAL_P95_Z)
            - base.pause_sigma_rel * float(np.mean([d.pause_exp.mean() for d in good])),
            0.0,
        ),
    }
    return base.model_copy(update={k: round(max(getattr(base, k), v), 3) for k, v in scale.items()})


def good_deviations(
    reference, audio_paths, sr: int = 16_000, half_window: int = 2
) -> list[Deviations]:
    """Deviation tables of good recordings (other speakers, same text) vs ``reference``."""
    from speechlens.analysis.compare import compare_to_reference
    from speechlens.audio import load_audio
    from speechlens.features import extract_speech_features

    out = []
    for path in audio_paths:
        x = load_audio(path, sr)
        par = extract_speech_features(x, sr, reference.aligner.align(x))
        out.append(compare_to_reference(reference.features, par, half_window))
    return out
