"""Chooses *where* flaws go: non-overlapping, linguistically sensible regions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from speechlens.acoustics.pitch import F0Range, f0_contour
from speechlens.dataset.flaws import FLAWS, FlawRequest, FlawType
from speechlens.schema import WordTiming
from speechlens.text import boundary_kind

MIN_VOICED_FOR_PITCH_FLAWS = 0.6


def word_voiced_fractions(
    source: np.ndarray, sr: int, words: Sequence[WordTiming], f0: F0Range
) -> list[float]:
    """Fraction of each word's frames that are voiced (pitch-bearing)."""
    times, contour = f0_contour(source, sr, f0.floor, f0.ceiling)
    voiced = ~np.isnan(contour)
    out = []
    for w in words:
        seg = voiced[np.searchsorted(times, w.start) : np.searchsorted(times, w.end)]
        out.append(float(seg.mean()) if len(seg) else 0.0)
    return out


class RegionPlanner:
    """Stateful planner: every accepted region blocks itself (plus a 1-word guard)."""

    def __init__(
        self,
        words: Sequence[WordTiming],
        rng: np.random.Generator,
        voiced: Sequence[float] | None = None,
    ):
        """``voiced`` (per-word voiced fraction) lets pitch flaws avoid words with no pitch."""
        self.n = len(words)
        self.rng = rng
        self.voiced = voiced
        self.free = np.ones(self.n, dtype=bool)
        self.boundary = [boundary_kind(w.punct_after) for w in words]
        self.gap_after = [
            words[i + 1].start - words[i].end if i + 1 < self.n else 0.0 for i in range(self.n)
        ]

    def _is_free(self, s: int, e: int) -> bool:
        return 0 <= s <= e < self.n and bool(self.free[max(0, s - 1) : min(self.n, e + 2)].all())

    def _candidates(self, ftype: FlawType, length: int) -> list[int]:
        """Valid start indices for a region of ``length`` words."""
        d = FLAWS[ftype]
        starts = range(0, self.n - length + 1)
        if d.anchor == "phrase":
            ok = [s for s in starts if self._is_free(s, s + length - 1)]
            preferred = [s for s in ok if s == 0 or self.boundary[s - 1] != "none"]
            return preferred if preferred and self.rng.random() < 0.7 else ok
        if d.anchor == "sentence_end":
            need_voiced = ftype == FlawType.UPTALK and self.voiced is not None
            return [
                s
                for s in starts
                if self.boundary[s + length - 1] == "sentence"
                and self._is_free(s, s + length - 1)
                and (not need_voiced or self.voiced[s + length - 1] >= MIN_VOICED_FOR_PITCH_FLAWS)
            ]
        wants_none = d.anchor == "boundary_none"
        return [
            s
            for s in starts
            if self._is_free(s, s + 1)
            and (self.boundary[s] == "none") == wants_none
            and self.gap_after[s] >= d.min_gap_s
        ]

    def plan(self, ftype: FlawType, severity: float) -> FlawRequest | None:
        lo, hi = FLAWS[ftype].span
        for length in range(int(self.rng.integers(lo, hi + 1)), lo - 1, -1):
            cands = self._candidates(ftype, length)
            if cands:
                s = int(self.rng.choice(cands))
                self.free[s : s + length] = False
                return FlawRequest(ftype, float(severity), s, s + length - 1)
        return None
