"""Rubric scoring: five delivery categories -> 0-100, plus speech-type weight presets.

Each category score is ``100 * exp(-gain * D)`` where ``D`` is the mean, over words, of the
deviation *beyond one natural-variation unit* (|z| - 1, capped). Free within tolerance, smooth
beyond it, and sensitive to *how much* of the speech deviates and *how far*, not just to
whether a region was flagged. Weights are transparent heuristics, not judge-calibrated.
"""

from __future__ import annotations

import numpy as np

from speechlens.analysis.models import Scores
from speechlens.config import DetectionConfig

CATEGORIES = ("pacing", "pauses", "pitch", "energy", "clarity")

PRESETS: dict[str, dict[str, float]] = {
    "balanced": dict.fromkeys(CATEGORIES, 0.2),
    "interpretive": {
        "pacing": 0.20,
        "pauses": 0.20,
        "pitch": 0.30,
        "energy": 0.15,
        "clarity": 0.15,
    },
    "declamation": {"pacing": 0.15, "pauses": 0.20, "pitch": 0.25, "energy": 0.25, "clarity": 0.15},
    "extemporaneous": {
        "pacing": 0.30,
        "pauses": 0.25,
        "pitch": 0.15,
        "energy": 0.15,
        "clarity": 0.15,
    },
    "persuasive": {"pacing": 0.20, "pauses": 0.25, "pitch": 0.20, "energy": 0.20, "clarity": 0.15},
}


# A pause (or a rising final word) is a *point event* that colours the surrounding phrase, yet it
# is measured on one word. Averaging it over every word would let a 2-second misplaced pause barely
# register, so point-event penalties are spread over ~5 words.
POINT_EVENT_SPREAD = 5.0


def _penalty(z: np.ndarray, cap: float, spread: float = 1.0) -> float:
    excess = np.minimum(np.maximum(np.abs(z) - 1.0, 0.0), cap) / cap
    return float(min(spread * np.mean(excess), 1.0))


def score_speech(z: dict[str, np.ndarray], preset: str, cfg: DetectionConfig) -> Scores:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    cap, gain = cfg.z_cap, cfg.score_gain
    flat = np.maximum(-z["pitch"], 0.0)  # compressed pitch range (sustained)
    rising = np.maximum(z["terminal"], 0.0)  # rising final words (point events)
    clarity = np.maximum(-z["clarity"], 0.0)  # only loss of clarity is a flaw
    pen = {
        "pacing": _penalty(z["pace"], cap),
        "pauses": _penalty(z["pause"], cap, POINT_EVENT_SPREAD),
        "pitch": min(_penalty(flat, cap) + _penalty(rising, cap, POINT_EVENT_SPREAD), 1.0),
        "energy": _penalty(z["energy"], cap),
        "clarity": _penalty(clarity, cap),
    }
    cats = {k: round(100.0 * float(np.exp(-gain * v)), 1) for k, v in pen.items()}
    w = PRESETS[preset]
    overall = round(sum(w[k] * cats[k] for k in CATEGORIES), 1)
    return Scores(overall=overall, categories=cats, preset=preset, weights=w)
