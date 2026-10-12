"""Refine word boundaries against the energy envelope.

DTW boundaries are only accurate to a few tens of ms, yet pause *duration* is the most
visible delivery cue. We therefore re-derive each inter-word gap from the signal: a gap is
the longest run of low-energy frames near the aligned boundary (or a single touching point
if there is none).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

from speechlens.schema import Alignment

MIN_PAUSE_S = 0.06
SEARCH_S = 0.12


def silence_mask(energy_db: np.ndarray, hop: float) -> np.ndarray:
    """Frames that are silent *relative to the local speech level*.

    A single global threshold would read quiet passages (e.g. a drop in volume) as silence and
    clip the words inside them. The threshold therefore follows the local peak level (30 dB
    below it) but never falls below 3 dB above the recording's noise floor.
    """
    floor = np.percentile(energy_db, 5)
    size = max(3, int(round(0.7 / hop)))
    local_peak = maximum_filter1d(energy_db, size=size, mode="nearest")
    return energy_db < np.maximum(floor + 3.0, local_peak - 30.0)


def refine_boundaries(alignment: Alignment, times: np.ndarray, energy_db: np.ndarray) -> Alignment:
    """Return a copy of ``alignment`` whose inter-word gaps follow the energy envelope."""
    words = list(alignment.words)
    if len(words) < 2:
        return alignment
    hop = float(np.median(np.diff(times)))
    silent = silence_mask(energy_db, hop)

    def runs(t0: float, t1: float) -> list[tuple[float, float]]:
        lo = max(int(np.searchsorted(times, t0)), 0)
        hi = min(int(np.searchsorted(times, t1)), len(times))
        out, start = [], None
        for k in range(lo, hi):
            if silent[k] and start is None:
                start = k
            if (not silent[k] or k == hi - 1) and start is not None:
                end = k if not silent[k] else k + 1
                out.append((times[start] - hop / 2, times[end - 1] + hop / 2))
                start = None
        return out

    new = [w.model_copy() for w in words]
    for i in range(len(words) - 1):
        a, b = new[i], new[i + 1]
        lo, hi = a.end - SEARCH_S, b.start + SEARCH_S
        cands = [r for r in runs(lo, hi) if r[1] - r[0] >= MIN_PAUSE_S]
        if cands:
            s, e = max(cands, key=lambda r: r[1] - r[0])
            a.end, b.start = max(s, a.start), min(e, b.end)
        else:  # no pause: a single cut at the energy minimum between the two words
            k0 = max(int(np.searchsorted(times, a.end - 0.05)), 0)
            k1 = min(int(np.searchsorted(times, b.start + 0.05)) + 1, len(times))
            if k1 > k0:
                cut = float(times[k0 + int(np.argmin(energy_db[k0:k1]))])
                cut = min(max(cut, a.start), b.end)
                a.end = b.start = cut
        if a.end > b.start:
            a.end = b.start = (a.end + b.start) / 2
    return alignment.model_copy(update={"words": new, "aligner": alignment.aligner + "+refined"})
