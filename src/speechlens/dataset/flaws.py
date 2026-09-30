"""The flaw library: ten parametric delivery flaws with a common severity scale.

Severity ``s`` in (0, 1] scales every flaw from barely perceptible to egregious.
Each injector edits segments of a :class:`Timeline`, and reports which segments
define the labelled region so ground truth can be read off the rendered audio.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from speechlens.acoustics.pitch import F0Range
from speechlens.audio import db_to_gain
from speechlens.dataset.dsp import (
    RoomTone,
    add_noise_at_snr,
    edit_region,
    extend_gap,
    lowpass_region,
    shrink_gap,
)
from speechlens.dataset.timeline import Timeline


class FlawType(str, Enum):
    RUSH = "rush"
    DRAG = "drag"
    MISPLACED_PAUSE = "misplaced_pause"
    MISSING_PAUSE = "missing_pause"
    OVERLONG_PAUSE = "overlong_pause"
    MONOTONE = "monotone"
    TRAIL_OFF = "trail_off"
    VOLUME_COLLAPSE = "volume_collapse"
    MUMBLE = "mumble"
    UPTALK = "uptalk"


@dataclass(frozen=True)
class FlawRequest:
    type: FlawType
    severity: float
    word_start: int  # inclusive
    word_end: int  # inclusive


@dataclass
class BuildContext:
    sr: int
    f0: F0Range
    tone: RoomTone
    context: int  # PSOLA context samples


@dataclass
class Applied:
    label_positions: list[int]  # segment positions spanning the labelled region
    focus_positions: list[int]  # segment positions of the manipulated element
    params: dict[str, Any]


Injector = Callable[[Timeline, BuildContext, FlawRequest, np.random.Generator], Applied]


def _words(req: FlawRequest) -> range:
    return range(req.word_start, req.word_end + 1)


def _word_span_positions(req: FlawRequest) -> list[int]:
    return [Timeline.word_pos(req.word_start), Timeline.word_pos(req.word_end)]


def _edit_words(tl: Timeline, ctx: BuildContext, req: FlawRequest, **kw: Any) -> None:
    for i in _words(req):
        seg = tl.word(i)
        seg.audio = edit_region(tl.source, ctx.sr, *seg.src, f0=ctx.f0, context=ctx.context, **kw)
        seg.modified = True


def _region_seconds(tl: Timeline, req: FlawRequest) -> tuple[float, float]:
    return tl.word(req.word_start).src[0] / tl.sr, tl.word(req.word_end).src[1] / tl.sr


# -- pacing ------------------------------------------------------------------------------
def _rush(tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator) -> Applied:
    rate = 1.0 + 0.9 * req.severity
    gap_scale = 1.0 - 0.7 * req.severity
    _edit_words(tl, ctx, req, duration_factor=1.0 / rate)
    for k in range(req.word_start + 1, req.word_end + 1):
        g = tl.gap(k)
        g.audio, g.modified = shrink_gap(g.audio, int(round(len(g.audio) * gap_scale))), True
    return Applied(
        _word_span_positions(req),
        _word_span_positions(req),
        {"rate_multiplier": round(rate, 3), "gap_scale": round(gap_scale, 3)},
    )


def _drag(tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator) -> Applied:
    factor = 1.0 + 1.0 * req.severity
    _edit_words(tl, ctx, req, duration_factor=factor)
    for k in range(req.word_start + 1, req.word_end + 1):
        g = tl.gap(k)
        extra = int(round(len(g.audio) * req.severity))
        g.audio, g.modified = extend_gap(g.audio, extra, ctx.tone, rng), True
    return Applied(
        _word_span_positions(req),
        _word_span_positions(req),
        {"rate_multiplier": round(1.0 / factor, 3)},
    )


# -- pauses (region = the two words around the boundary + the pause itself) ---------------
def _pause_positions(req: FlawRequest) -> tuple[list[int], list[int]]:
    k = req.word_start + 1
    return (
        [Timeline.word_pos(req.word_start), Timeline.word_pos(req.word_end)],
        [Timeline.gap_pos(k)],
    )


def _misplaced_pause(
    tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator
) -> Applied:
    extra_s = 0.25 + 1.35 * req.severity
    g = tl.gap(req.word_start + 1)
    original = len(g.audio) / tl.sr
    g.audio, g.modified = extend_gap(g.audio, int(round(extra_s * tl.sr)), ctx.tone, rng), True
    label, focus = _pause_positions(req)
    return Applied(
        label,
        focus,
        {"original_pause_s": round(original, 3), "inserted_pause_s": round(extra_s, 3)},
    )


def _overlong_pause(
    tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator
) -> Applied:
    extra_s = 0.40 + 2.00 * req.severity
    g = tl.gap(req.word_start + 1)
    original = len(g.audio) / tl.sr
    g.audio, g.modified = extend_gap(g.audio, int(round(extra_s * tl.sr)), ctx.tone, rng), True
    label, focus = _pause_positions(req)
    return Applied(
        label, focus, {"original_pause_s": round(original, 3), "added_pause_s": round(extra_s, 3)}
    )


def _missing_pause(
    tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator
) -> Applied:
    g = tl.gap(req.word_start + 1)
    original = len(g.audio) / tl.sr
    g.audio, g.modified = shrink_gap(g.audio, int(round(len(g.audio) * (1.0 - req.severity)))), True
    label, focus = _pause_positions(req)
    return Applied(
        label,
        focus,
        {"original_pause_s": round(original, 3), "new_pause_s": round(len(g.audio) / tl.sr, 3)},
    )


# -- pitch ------------------------------------------------------------------------------
def _monotone(
    tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator
) -> Applied:
    med, keep = ctx.f0.median, 1.0 - req.severity

    def compress(_t: np.ndarray, f0: np.ndarray) -> np.ndarray:
        return med * np.exp2(keep * np.log2(f0 / med))

    _edit_words(tl, ctx, req, pitch_edit=compress)
    return Applied(
        _word_span_positions(req),
        _word_span_positions(req),
        {"pitch_range_retained": round(keep, 3)},
    )


def _uptalk(tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator) -> Applied:
    t0, t1 = _region_seconds(tl, req)
    semitones = 7.0 * req.severity

    def rise(t: np.ndarray, f0: np.ndarray) -> np.ndarray:
        u = np.clip((t - t0) / max(t1 - t0, 1e-6), 0.0, 1.0) ** 2
        return f0 * np.exp2(semitones * u / 12.0)

    _edit_words(tl, ctx, req, pitch_edit=rise)
    return Applied(
        _word_span_positions(req),
        _word_span_positions(req),
        {"terminal_rise_semitones": round(semitones, 2)},
    )


# -- energy / clarity -------------------------------------------------------------------
def _apply_gain(
    tl: Timeline, req: FlawRequest, gain_at: Callable[[np.ndarray], np.ndarray]
) -> None:
    for i in _words(req):
        seg = tl.word(i)
        t = (seg.src[0] + np.arange(seg.src[1] - seg.src[0])) / tl.sr
        seg.audio, seg.modified = tl.source[seg.src[0] : seg.src[1]] * gain_at(t), True


def _trail_off(
    tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator
) -> Applied:
    t0, t1 = _region_seconds(tl, req)
    drop_db = 6.0 + 18.0 * req.severity
    _apply_gain(
        tl, req, lambda t: db_to_gain(-drop_db * np.clip((t - t0) / max(t1 - t0, 1e-6), 0, 1))
    )
    return Applied(
        _word_span_positions(req), _word_span_positions(req), {"final_drop_db": round(drop_db, 2)}
    )


def _volume_collapse(
    tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator
) -> Applied:
    drop_db = 8.0 + 16.0 * req.severity
    _apply_gain(tl, req, lambda t: np.full_like(t, db_to_gain(-drop_db)))
    return Applied(
        _word_span_positions(req), _word_span_positions(req), {"drop_db": round(drop_db, 2)}
    )


def _mumble(tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator) -> Applied:
    cutoff = 6500.0 - 5300.0 * req.severity
    snr = 35.0 - 20.0 * req.severity
    for i in _words(req):
        seg = tl.word(i)
        y = lowpass_region(tl.source, ctx.sr, *seg.src, cutoff, ctx.context)
        seg.audio, seg.modified = (
            np.clip(add_noise_at_snr(y, snr, rng, ctx.sr, cutoff), -1.0, 1.0),
            True,
        )
    return Applied(
        _word_span_positions(req),
        _word_span_positions(req),
        {"lowpass_hz": round(cutoff), "snr_db": round(snr, 1)},
    )


@dataclass(frozen=True)
class FlawDef:
    type: FlawType
    apply: Injector
    span: tuple[int, int]  # allowed word-span length
    anchor: str  # phrase | sentence_end | boundary_none | boundary_punct
    signature: str  # expected acoustic signature (used by the explanation engine)
    description: str
    min_gap_s: float = 0.0  # required pre-existing pause (pause-removal needs something to remove)


FLAWS: dict[FlawType, FlawDef] = {
    d.type: d
    for d in [
        FlawDef(
            FlawType.RUSH,
            _rush,
            (6, 12),
            "phrase",
            "local speech rate up, pauses compressed",
            "Speaking too fast over a phrase, squeezing out pauses.",
        ),
        FlawDef(
            FlawType.DRAG,
            _drag,
            (4, 8),
            "phrase",
            "local speech rate down, pauses stretched",
            "Dragging: unnaturally slow delivery over a phrase.",
        ),
        FlawDef(
            FlawType.MISPLACED_PAUSE,
            _misplaced_pause,
            (2, 2),
            "boundary_none",
            "long silence where the text has no punctuation",
            "A pause in the middle of a phrase where the text does not call for one.",
        ),
        FlawDef(
            FlawType.MISSING_PAUSE,
            _missing_pause,
            (2, 2),
            "boundary_punct",
            "silence at punctuation collapsed",
            "Running through a comma or full stop without pausing.",
            min_gap_s=0.15,
        ),
        FlawDef(
            FlawType.OVERLONG_PAUSE,
            _overlong_pause,
            (2, 2),
            "boundary_punct",
            "silence at punctuation far above expectation",
            "A pause that hangs far too long.",
        ),
        FlawDef(
            FlawType.MONOTONE,
            _monotone,
            (8, 16),
            "phrase",
            "F0 range compressed toward the median",
            "Flat, monotone pitch with little melodic variation.",
        ),
        FlawDef(
            FlawType.TRAIL_OFF,
            _trail_off,
            (3, 5),
            "sentence_end",
            "energy decays toward the phrase end",
            "Energy fades away at the end of a sentence.",
        ),
        FlawDef(
            FlawType.VOLUME_COLLAPSE,
            _volume_collapse,
            (4, 8),
            "phrase",
            "sustained energy drop",
            "Volume drops sharply for a stretch of speech.",
        ),
        FlawDef(
            FlawType.MUMBLE,
            _mumble,
            (3, 6),
            "phrase",
            "high-frequency loss, lower SNR",
            "Mumbled, muffled articulation with reduced clarity.",
        ),
        FlawDef(
            FlawType.UPTALK,
            _uptalk,
            (1, 2),
            "sentence_end",
            "F0 rises at the end of a statement",
            "Rising terminal pitch that makes statements sound like questions.",
        ),
    ]
}


def inject(tl: Timeline, ctx: BuildContext, req: FlawRequest, rng: np.random.Generator) -> Applied:
    """Apply one flaw to ``tl`` in place."""
    return FLAWS[req.type].apply(tl, ctx, req, rng)
