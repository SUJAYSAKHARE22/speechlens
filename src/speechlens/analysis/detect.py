"""Temporal grounding: turn per-word deviation tracks into flaw regions.

Each detector looks at one deviation track (in z-units), finds runs of consecutive words that
exceed the threshold, and reports the word span. Region times come from the participant's own
refined word alignment, so start/end are word-boundary precise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from speechlens.analysis.compare import Deviations
from speechlens.config import DetectionConfig
from speechlens.features import WordFeatures

FAMILY = {
    "rush": "pace", "drag": "pace",
    "misplaced_pause": "pause", "missing_pause": "pause", "overlong_pause": "pause",
    "monotone": "pitch", "uptalk": "pitch",
    "trail_off": "energy", "volume_collapse": "energy",
    "mumble": "clarity",
}  # fmt: skip


@dataclass
class RawRegion:
    type: str
    i0: int  # first word (inclusive)
    i1: int  # last word (inclusive)
    peak_z: float
    focus_after: int | None = None  # pause flaws: the pause follows this word


def _runs(mask: np.ndarray, max_gap: int, min_len: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    runs, start, prev = [], idx[0], idx[0]
    for k in idx[1:]:
        if k - prev - 1 > max_gap:
            runs.append((int(start), int(prev)))
            start = k
        prev = k
    runs.append((int(start), int(prev)))
    return [(a, b) for a, b in runs if b - a + 1 >= min_len]


def smoothed_pace(z: dict[str, np.ndarray]) -> np.ndarray:
    """3-word median of the pace z-scores (what the pacing detector actually thresholds)."""
    return median_filter(z["pace"], size=3, mode="nearest")


def _sentence_ends(words: list[WordFeatures]) -> list[int]:
    ends = [w.index for w in words if w.boundary == "sentence"]
    return ends if ends and ends[-1] == len(words) - 1 else [*ends, len(words) - 1]


def detect_flaws(
    dev: Deviations, z: dict[str, np.ndarray], words: list[WordFeatures], cfg: DetectionConfig
) -> list[RawRegion]:
    thr, n = cfg.z_threshold, len(words)
    out: list[RawRegion] = []
    taken = np.zeros(n, dtype=bool)  # words already explained by a more specific energy flaw

    # -- energy fading away at a sentence end (checked before generic volume loss) ----------
    for e in _sentence_ends(words):
        lo = max(0, e - cfg.trail_window + 1)
        ze = z["energy"][lo : e + 1]
        if len(ze) < 3:
            continue
        peak = int(np.argmax(ze[:-1]))
        decline = ze[peak] - ze[-1]
        if ze[-1] > -thr or decline < 0.8 * thr:
            continue
        tail = ze[peak:]  # only now is a steadiness check meaningful (tail is non-constant)
        steady = (
            float(np.corrcoef(np.arange(len(tail)), tail)[0, 1])
            if len(tail) > 2 and np.std(tail) > 1e-9
            else 0.0
        )
        if steady < -0.6:
            out.append(RawRegion("trail_off", lo + peak, e, float(-ze[-1])))
            taken[lo + peak : e + 1] = True

    # -- sustained loss of volume ------------------------------------------------------------
    ze_s = median_filter(z["energy"], size=3, mode="nearest")
    for a, b in _runs((ze_s <= -thr) & ~taken, cfg.max_gap_words, cfg.min_words_energy):
        out.append(RawRegion("volume_collapse", a, b, float(-ze_s[a : b + 1].min())))

    # -- pacing -----------------------------------------------------------------------------
    zp = smoothed_pace(z)
    for a, b in _runs(zp <= -thr, cfg.max_gap_words, cfg.min_words_pace):
        out.append(RawRegion("rush", a, b, float(-zp[a : b + 1].min())))
    for a, b in _runs(zp >= thr, cfg.max_gap_words, cfg.min_words_pace):
        out.append(RawRegion("drag", a, b, float(zp[a : b + 1].max())))

    # -- pauses (judged against the punctuation the text provides) --------------------------------
    for i in range(n - 1):
        obs, exp = dev.pause_obs[i], dev.pause_exp[i]
        if z["pause"][i] >= thr and obs - exp >= cfg.pause_min_excess_s:
            kind = "misplaced_pause" if words[i].boundary == "none" else "overlong_pause"
            out.append(RawRegion(kind, i, i + 1, float(z["pause"][i]), focus_after=i))
        elif (
            words[i].boundary != "none"
            and exp >= cfg.missing_pause_min_expected_s
            and obs <= cfg.missing_pause_ratio * exp
            and exp - obs >= cfg.missing_pause_min_loss_s
        ):
            sigma_equiv = (exp - obs) / max(exp, 1e-6) * thr / (1 - cfg.missing_pause_ratio)
            out.append(RawRegion("missing_pause", i, i + 1, float(sigma_equiv), focus_after=i))

    # -- pitch ---------------------------------------------------------------------------------
    for a, b in _runs(z["pitch"] <= -thr, cfg.max_gap_words, cfg.min_words_pitch):
        out.append(RawRegion("monotone", a, b, float(-z["pitch"][a : b + 1].min())))
    for e in _sentence_ends(words):
        if z["terminal"][e] >= thr:
            out.append(RawRegion("uptalk", e, e, float(z["terminal"][e])))

    # -- clarity ---------------------------------------------------------------------------------
    for a, b in _runs(z["clarity"] <= -thr, cfg.max_gap_words, cfg.min_words_clarity):
        out.append(RawRegion("mumble", a, b, float(-z["clarity"][a : b + 1].min())))

    return _consolidate(sorted(out, key=lambda r: (r.i0, r.i1)))


def _consolidate(regions: list[RawRegion]) -> list[RawRegion]:
    """Drop pause regions already explained by a pacing region that contains them.

    Rushing squeezes the pauses inside the phrase (and dragging stretches them); the pacing
    explanation says so, so reporting each squeezed pause again would double-count one flaw.
    """
    pace = [r for r in regions if r.type in ("rush", "drag")]

    def explained(r: RawRegion) -> bool:
        cause = {"missing_pause": "rush", "overlong_pause": "drag"}.get(r.type)
        return cause is not None and any(
            p.type == cause and p.i0 <= r.i0 and r.i1 <= p.i1 for p in pace
        )

    return [r for r in regions if not explained(r)]


def severity_from_z(peak_z: float, thr: float) -> float:
    """Monotone map of threshold exceedance to [0.2, 1]: z = thr -> 0.2, z = 4*thr -> 1."""
    return float(0.2 + 0.8 * np.clip((peak_z - thr) / (3 * thr), 0, 1))


def confidence_from_z(peak_z: float, thr: float) -> float:
    """Heuristic confidence: grows smoothly from 0.5 at the threshold towards 1."""
    return float(1 - 0.5 * np.exp(-(peak_z - thr) / thr))
