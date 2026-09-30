"""Segment timeline: the structure that keeps ground-truth labels exact.

A recording is cut into alternating *gap* and *word* segments using the forced
alignment:  gap0, word0, gap1, word1, ..., word{N-1}, gap{N}.  Gap ``k`` is the
silence *before* word ``k``; gap ``N`` is the trailing silence.

Flaw injectors replace the audio of individual segments (changing their length).
Rendering concatenates the segments and records where each one lands, so the label
of every flaw is derived from the *final* audio rather than predicted. Untouched
segments are contiguous slices of the source, hence bit-exact after rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from speechlens.audio import fade_edges, ms_to_samples
from speechlens.schema import Alignment, WordTiming


@dataclass
class Segment:
    kind: str  # "gap" | "word"
    index: int  # word index (gap k precedes word k)
    src: tuple[int, int]  # sample range in the source recording
    audio: np.ndarray
    modified: bool = False


@dataclass
class Rendered:
    audio: np.ndarray
    seg_start: np.ndarray  # first sample of each segment in the output
    seg_end: np.ndarray


def _snap(env: np.ndarray, pos: int, radius: int) -> int:
    lo, hi = max(0, pos - radius), min(len(env), pos + radius + 1)
    return lo + int(np.argmin(env[lo:hi])) if hi > lo else pos


@dataclass
class Timeline:
    source: np.ndarray
    sr: int
    words: list[WordTiming]
    segments: list[Segment] = field(default_factory=list)
    fade_samples: int = 48

    @classmethod
    def from_alignment(
        cls,
        source: np.ndarray,
        sr: int,
        alignment: Alignment,
        snap_ms: float = 25.0,
        fade_ms: float = 3.0,
    ) -> Timeline:
        n_src = len(source)
        win = max(1, ms_to_samples(5.0, sr))
        env = np.sqrt(np.convolve(source**2, np.ones(win) / win, mode="same"))
        radius = ms_to_samples(snap_ms, sr)

        starts = [int(round(w.start * sr)) for w in alignment.words]
        ends = [int(round(w.end * sr)) for w in alignment.words]
        touch = ms_to_samples(10.0, sr)
        for i in range(1, len(starts)):  # words that touch share a single cut point
            if starts[i] - ends[i - 1] <= touch:
                starts[i] = ends[i - 1] = (starts[i] + ends[i - 1]) // 2

        # Snap each cut to the local energy minimum, then enforce monotonicity.
        flat = np.array([p for pair in zip(starts, ends, strict=True) for p in pair], dtype=int)
        flat = np.array([_snap(env, int(p), radius) for p in flat])
        flat = np.clip(np.maximum.accumulate(flat), 0, n_src)

        tl = cls(
            source=source,
            sr=sr,
            words=list(alignment.words),
            fade_samples=ms_to_samples(fade_ms, sr),
        )
        prev = 0
        for i in range(len(alignment.words)):
            s, e = int(flat[2 * i]), int(flat[2 * i + 1])
            tl.segments.append(Segment("gap", i, (prev, s), source[prev:s].copy()))
            tl.segments.append(Segment("word", i, (s, e), source[s:e].copy()))
            prev = e
        n = len(alignment.words)
        tl.segments.append(Segment("gap", n, (prev, n_src), source[prev:n_src].copy()))
        return tl

    # -- accessors ---------------------------------------------------------------------
    @staticmethod
    def word_pos(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def gap_pos(k: int) -> int:
        return 2 * k

    def word(self, i: int) -> Segment:
        return self.segments[self.word_pos(i)]

    def gap(self, k: int) -> Segment:
        return self.segments[self.gap_pos(k)]

    # -- rendering ---------------------------------------------------------------------
    def render(self) -> Rendered:
        parts, starts, ends, pos = [], [], [], 0
        for seg in self.segments:
            a = (
                fade_edges(seg.audio, self.fade_samples)
                if seg.modified and seg.kind == "word"
                else seg.audio
            )
            starts.append(pos)
            pos += len(a)
            ends.append(pos)
            parts.append(a)
        return Rendered(np.concatenate(parts), np.array(starts), np.array(ends))

    def interval(self, rendered: Rendered, positions: list[int]) -> tuple[float, float]:
        """Time span (s) covered by the given segment positions in the rendered audio."""
        lo, hi = min(positions), max(positions)
        return rendered.seg_start[lo] / self.sr, rendered.seg_end[hi] / self.sr

    def output_alignment(self, rendered: Rendered, aligner: str = "timeline") -> Alignment:
        """Word alignment of the rendered audio (exact by construction)."""
        words = [
            WordTiming(
                index=w.index,
                text=w.text,
                start=float(rendered.seg_start[self.word_pos(i)] / self.sr),
                end=float(rendered.seg_end[self.word_pos(i)] / self.sr),
                punct_after=w.punct_after,
            )
            for i, w in enumerate(self.words)
        ]
        return Alignment(
            words=words,
            sample_rate=self.sr,
            duration=float(len(rendered.audio) / self.sr),
            aligner=aligner,
        )
