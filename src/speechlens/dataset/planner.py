"""Chooses *where* flaws go: non-overlapping, linguistically sensible regions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from speechlens.dataset.flaws import FLAWS, FlawRequest, FlawType
from speechlens.schema import WordTiming
from speechlens.text import boundary_kind


class RegionPlanner:
    """Stateful planner: every accepted region blocks itself (plus a 1-word guard)."""

    def __init__(self, words: Sequence[WordTiming], rng: np.random.Generator):
        self.n = len(words)
        self.rng = rng
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
            return [
                s
                for s in starts
                if self.boundary[s + length - 1] == "sentence" and self._is_free(s, s + length - 1)
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
