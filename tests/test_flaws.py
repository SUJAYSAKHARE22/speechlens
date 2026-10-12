"""Each flaw must (1) be labelled exactly, (2) leave everything outside its region untouched,
and (3) actually change the acoustic property it claims to change."""

import numpy as np
import pytest

from speechlens.acoustics.pitch import f0_contour
from speechlens.audio import rms_db
from speechlens.dataset.flaws import FlawRequest, FlawType, inject
from speechlens.dataset.planner import RegionPlanner
from speechlens.dataset.timeline import Timeline
from tests.conftest import SR


def _run(source, alignment, ctx, ftype, severity, seed=0, voiced=None):
    req = RegionPlanner(alignment.words, np.random.default_rng(seed), voiced).plan(ftype, severity)
    assert req is not None
    tl = Timeline.from_alignment(source, SR, alignment)
    base = Timeline.from_alignment(source, SR, alignment)
    applied = inject(tl, ctx, req, np.random.default_rng(seed))
    rendered = tl.render()
    return req, tl, base, applied, rendered


def _slice(x, a, b):
    return x[int(round(a * SR)) : int(round(b * SR))]


def _base_span(base, req):
    return base.word(req.word_start).src[0] / SR, base.word(req.word_end).src[1] / SR


@pytest.mark.parametrize("ftype", list(FlawType))
def test_labels_exact_and_outside_region_untouched(source, alignment, ctx, ftype):
    req, tl, base, ap, r = _run(source, alignment, ctx, ftype, 0.7)
    start, end = tl.interval(r, ap.label_positions)
    fstart, fend = tl.interval(r, ap.focus_positions)
    assert 0 <= start < end <= len(r.audio) / SR
    assert start - 1e-9 <= fstart < fend <= end + 1e-9

    mods = [i for i, s in enumerate(tl.segments) if s.modified]
    first, last = min(mods), max(mods)
    n_pre = int(r.seg_start[first])
    assert np.array_equal(r.audio[:n_pre], source[:n_pre])  # prefix bit-exact
    n_post = len(r.audio) - int(r.seg_end[last])
    assert np.array_equal(r.audio[len(r.audio) - n_post :], source[len(source) - n_post :])
    tl.output_alignment(r)  # validates ordering / non-overlap


def test_rush_speeds_up_by_the_stated_factor(source, alignment, ctx):
    req, tl, base, ap, r = _run(source, alignment, ctx, FlawType.RUSH, 0.8)
    a, b = tl.interval(r, ap.label_positions)
    a0, b0 = _base_span(base, req)
    # region contains inter-word gaps that shrink even more, so duration falls at least this much
    assert (b - a) <= (b0 - a0) / ap.params["rate_multiplier"] * 1.05


def test_drag_slows_down_by_the_stated_factor(source, alignment, ctx):
    req, tl, base, ap, r = _run(source, alignment, ctx, FlawType.DRAG, 0.8)
    a, b = tl.interval(r, ap.label_positions)
    a0, b0 = _base_span(base, req)
    assert (b - a) >= (b0 - a0) * 1.8 * 0.95


def test_misplaced_and_overlong_pauses_lengthen_the_gap(source, alignment, ctx):
    for ftype, key in (
        (FlawType.MISPLACED_PAUSE, "inserted_pause_s"),
        (FlawType.OVERLONG_PAUSE, "added_pause_s"),
    ):
        req, tl, base, ap, r = _run(source, alignment, ctx, ftype, 0.6)
        fs, fe = tl.interval(r, ap.focus_positions)
        assert fe - fs == pytest.approx(ap.params["original_pause_s"] + ap.params[key], abs=0.01)


def test_missing_pause_removes_silence(source, alignment, ctx):
    req, tl, base, ap, r = _run(source, alignment, ctx, FlawType.MISSING_PAUSE, 0.9)
    fs, fe = tl.interval(r, ap.focus_positions)
    assert ap.params["original_pause_s"] >= 0.15
    assert fe - fs == pytest.approx(ap.params["original_pause_s"] * 0.1, abs=0.01)


def _st_std(x, ctx):
    _, f = f0_contour(x, SR, ctx.f0.floor, ctx.f0.ceiling)
    f = f[~np.isnan(f)]
    return float(np.std(12 * np.log2(f / np.median(f))))


def test_monotone_compresses_pitch_range_and_scales_with_severity(source, alignment, ctx):
    stds = {}
    for sev in (0.0, 0.4, 0.8):
        if sev == 0.0:
            req, tl, base, ap, r = _run(source, alignment, ctx, FlawType.MONOTONE, 0.4)
            stds[sev] = _st_std(_slice(source, *_base_span(base, req)), ctx)
        else:
            req, tl, base, ap, r = _run(source, alignment, ctx, FlawType.MONOTONE, sev)
            stds[sev] = _st_std(_slice(r.audio, *tl.interval(r, ap.label_positions)), ctx)
    assert stds[0.0] > stds[0.4] > stds[0.8]
    assert stds[0.8] < 0.35 * stds[0.0]


def test_volume_collapse_drops_level_by_the_stated_db(source, alignment, ctx):
    req, tl, base, ap, r = _run(source, alignment, ctx, FlawType.VOLUME_COLLAPSE, 0.5)
    before = rms_db(_slice(source, *_base_span(base, req)))
    after = rms_db(_slice(r.audio, *tl.interval(r, ap.label_positions)))
    assert before - after == pytest.approx(ap.params["drop_db"], abs=1.0)


def test_trail_off_energy_decays_towards_the_end(source, alignment, ctx):
    req, tl, base, ap, r = _run(source, alignment, ctx, FlawType.TRAIL_OFF, 0.8)
    last = tl.output_alignment(r).words[req.word_end]
    first = tl.output_alignment(r).words[req.word_start]
    drop = rms_db(_slice(r.audio, first.start, first.end)) - rms_db(
        _slice(r.audio, last.start, last.end)
    )
    drop0 = rms_db(
        _slice(source, tl.word(req.word_start).src[0] / SR, tl.word(req.word_start).src[1] / SR)
    ) - rms_db(_slice(source, tl.word(req.word_end).src[0] / SR, tl.word(req.word_end).src[1] / SR))
    assert drop - drop0 > 8.0


def test_mumble_removes_high_frequency_energy(source, alignment, ctx):
    def hf_share(x):
        p = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SR)
        return 10 * np.log10(p[f > 2500].sum() / p.sum())

    req, tl, base, ap, r = _run(source, alignment, ctx, FlawType.MUMBLE, 0.8)
    assert (
        hf_share(_slice(source, *_base_span(base, req)))
        - hf_share(_slice(r.audio, *tl.interval(r, ap.label_positions)))
        > 3.0
    )


def test_uptalk_raises_terminal_pitch(source, alignment, ctx, voiced):
    req, tl, base, ap, r = _run(source, alignment, ctx, FlawType.UPTALK, 0.9, voiced=voiced)
    assert voiced[req.word_end] >= 0.6  # the planner only picks words that can carry a rise

    def tail_st(x):
        _, f = f0_contour(x, SR, ctx.f0.floor, ctx.f0.ceiling)
        f = f[~np.isnan(f)]
        return 12 * np.log2(np.median(f[-max(3, len(f) // 4) :]) / ctx.f0.median)

    last = tl.output_alignment(r).words[req.word_end]
    lb = tl.word(req.word_end)
    assert (
        tail_st(_slice(r.audio, last.start, last.end))
        - tail_st(_slice(source, lb.src[0] / SR, lb.src[1] / SR))
        > 2.0
    )


def test_severity_is_monotone_in_effect_for_pacing(source, alignment, ctx):
    durs = []
    for sev in (0.2, 0.5, 0.9):
        req, tl, base, ap, r = _run(source, alignment, ctx, FlawType.RUSH, sev, seed=2)
        a, b = tl.interval(r, ap.label_positions)
        durs.append(b - a)
    assert durs[0] > durs[1] > durs[2]


def test_flawrequest_is_hashable_and_frozen():
    req = FlawRequest(FlawType.RUSH, 0.5, 1, 4)
    assert hash(req) == hash(FlawRequest(FlawType.RUSH, 0.5, 1, 4))
