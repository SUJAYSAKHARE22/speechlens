"""Per-word deviation of a participant from an expectation.

Two modes produce the *same* table, so detection/explanation/scoring are shared:

* **reference** (paired baseline of the same text): expectation = the baseline's value for the
  same word, after removing global differences (tempo; levels are already normalised).
* **norm** (reference-free fallback): expectation = speaker-relative norms (own median level,
  own articulation rate, punctuation-class pause norms, population pitch variability).

Conventions: ``pace`` and ``pitch`` are log2 ratios, ``terminal``/``energy``/``clarity`` are
differences, pauses are seconds.  Positive pace = slower than expected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from speechlens.config import Tolerance
from speechlens.features import SpeechFeatures


@dataclass
class NormModel:
    """Reference-free expectations (fit from good recordings, or sensible defaults)."""

    pause_by_boundary: dict[str, float] = field(
        default_factory=lambda: {"none": 0.05, "clause": 0.30, "sentence": 0.60}
    )
    pitch_std_st: float = 2.5
    terminal_st: float = -2.5

    @classmethod
    def fit(cls, feats: list[SpeechFeatures]) -> NormModel:
        pauses: dict[str, list[float]] = {"none": [], "clause": [], "sentence": []}
        stds, tails = [], []
        for f in feats:
            for w in f.words[:-1]:
                pauses[w.boundary].append(w.pause_after)
            for k in range(0, len(f.words) - 5, 5):
                v = f.frames.f0_st[f.frames.slice(f.words[k].start, f.words[k + 4].end)]
                v = v[~np.isnan(v)]
                if len(v) > 30:
                    stds.append(float(v.std()))
            tails += [
                w.f0_tail_st
                for w in f.words
                if w.boundary == "sentence" and w.f0_tail_st == w.f0_tail_st
            ]
        d = cls()
        return cls(
            pause_by_boundary={
                k: float(np.median(v)) if v else d.pause_by_boundary[k] for k, v in pauses.items()
            },
            pitch_std_st=float(np.median(stds)) if stds else d.pitch_std_st,
            terminal_st=float(np.median(tails)) if tails else d.terminal_st,
        )


@dataclass
class Deviations:
    mode: str
    tempo: float | None
    pace: np.ndarray
    pause_obs: np.ndarray
    pause_exp: np.ndarray
    pitch: np.ndarray
    terminal: np.ndarray
    energy: np.ndarray
    clarity: np.ndarray
    detail: dict[str, np.ndarray]


def _nz(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _window_std(f: SpeechFeatures, i0: int, i1: int) -> float:
    v = f.frames.f0_st[f.frames.slice(f.words[i0].start, f.words[i1].end)]
    v = v[~np.isnan(v)]
    return float(v.std()) if len(v) >= 15 else float("nan")


MIN_VOICED_FRAC = 0.5
MIN_VOICED_S = 0.10


def _reliable_pitch(f: SpeechFeatures) -> np.ndarray:
    """Words with enough voiced signal for a trustworthy terminal-pitch estimate."""
    return np.array(
        [
            w.voiced_frac >= MIN_VOICED_FRAC and w.duration * w.voiced_frac >= MIN_VOICED_S
            for w in f.words
        ]
    )


def _centred(vals: np.ndarray) -> np.ndarray:
    return vals - np.nanmedian(vals)


def compare_to_reference(
    ref: SpeechFeatures, par: SpeechFeatures, half_window: int = 2
) -> Deviations:
    if len(ref.words) != len(par.words):
        raise ValueError("reference and participant must contain the same words")
    n = len(par.words)
    d_p = np.array([w.duration for w in par.words])
    d_r = np.array([w.duration for w in ref.words])
    ok = (d_r >= 0.08) & (d_p > 0.0)
    ratio = np.where(ok, d_p / np.maximum(d_r, 1e-9), np.nan)
    tempo = float(np.nanmedian(ratio)) if ok.any() else 1.0
    pace = _nz(np.log2(ratio / tempo))

    pitch = np.zeros(n)
    sd_p, sd_r = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half_window), min(n - 1, i + half_window)
        sd_p[i], sd_r[i] = _window_std(par, lo, hi), _window_std(ref, lo, hi)
    pitch = _nz(np.log2(sd_p / sd_r))

    tail_p = np.array([w.f0_tail_st for w in par.words])
    tail_r = np.array([w.f0_tail_st for w in ref.words])
    final = np.array([w.boundary == "sentence" for w in par.words])
    final[-1] = True
    usable = final & _reliable_pitch(par) & _reliable_pitch(ref)
    terminal = _nz(np.where(usable, tail_p - tail_r, 0.0))

    e_p = np.array([w.energy_db for w in par.words])
    e_r = np.array([w.energy_db for w in ref.words])
    h_p = _centred(np.array([w.hf_db for w in par.words]))
    h_r = _centred(np.array([w.hf_db for w in ref.words]))
    return Deviations(
        mode="reference",
        tempo=tempo,
        pace=pace,
        pause_obs=np.array([w.pause_after for w in par.words]),
        pause_exp=tempo * np.array([w.pause_after for w in ref.words]),
        pitch=pitch,
        terminal=terminal,
        energy=_nz(e_p - e_r),
        clarity=_nz(h_p - h_r),
        detail={
            "obs_duration": d_p, "exp_duration": tempo * d_r,
            "pitch_obs_std": sd_p, "pitch_exp_std": sd_r,
            "tail_obs": tail_p, "tail_exp": tail_r,
            "energy_obs": e_p, "energy_exp": e_r,
            "hf_obs": h_p, "hf_exp": h_r,
        },
    )  # fmt: skip


def compare_to_norms(par: SpeechFeatures, norms: NormModel, half_window: int = 2) -> Deviations:
    n = len(par.words)
    d_p = np.array([w.duration for w in par.words])
    syl = np.array([w.n_syll for w in par.words], dtype=float)
    global_s = max(par.stats.articulation_s_per_syll, 1e-6)
    local = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        local[i] = d_p[lo:hi].sum() / max(syl[lo:hi].sum(), 1.0)
    pace = np.log2(local / global_s)

    sd_p = np.array(
        [_window_std(par, max(0, i - half_window), min(n - 1, i + half_window)) for i in range(n)]
    )
    final = np.array([w.boundary == "sentence" for w in par.words])
    final[-1] = True
    tail = np.array([w.f0_tail_st for w in par.words])
    e_p = np.array([w.energy_db for w in par.words])
    h_p = _centred(np.array([w.hf_db for w in par.words]))
    pause_exp = np.array([norms.pause_by_boundary[w.boundary] for w in par.words])
    return Deviations(
        mode="norm",
        tempo=None,
        pace=_nz(pace),
        pause_obs=np.array([w.pause_after for w in par.words]),
        pause_exp=pause_exp,
        pitch=_nz(np.log2(sd_p / norms.pitch_std_st)),
        terminal=_nz(np.where(final & _reliable_pitch(par), tail - norms.terminal_st, 0.0)),
        energy=_nz(e_p - np.nanmedian(e_p)),
        clarity=_nz(h_p),
        detail={
            "obs_duration": d_p, "exp_duration": syl * global_s,
            "pitch_obs_std": sd_p, "pitch_exp_std": np.full(n, norms.pitch_std_st),
            "tail_obs": tail, "tail_exp": np.full(n, norms.terminal_st),
            "energy_obs": e_p, "energy_exp": np.full(n, np.nanmedian(e_p)),
            "hf_obs": h_p, "hf_exp": np.zeros(n),
        },
    )  # fmt: skip


def z_scores(dev: Deviations, tol: Tolerance) -> dict[str, np.ndarray]:
    """Scale every deviation by the natural between-speaker variation of its metric."""
    pause_sigma = tol.pause_sigma_abs + tol.pause_sigma_rel * dev.pause_exp
    z_pause = (dev.pause_obs - dev.pause_exp) / pause_sigma
    z_pause[-1] = 0.0  # the trailing silence after the final word is not a pause
    return {
        "pace": dev.pace / tol.pace_sigma,
        "pause": z_pause,
        "pitch": dev.pitch / tol.pitch_sigma,
        "terminal": dev.terminal / tol.terminal_sigma,
        "energy": dev.energy / tol.energy_sigma,
        "clarity": dev.clarity / tol.clarity_sigma,
    }
