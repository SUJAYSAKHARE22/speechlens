"""Regression tests for the DSP layer, chiefly determinism of Praat resynthesis."""

import numpy as np
import pytest

from speechlens.dataset.dsp import RoomTone, edit_region, extend_gap, shrink_gap
from tests.conftest import SR


@pytest.mark.parametrize("factor", [0.55, 1.0, 1.8])
def test_psola_edit_is_deterministic_and_exact_length(source, ctx, factor):
    a, b = 2 * SR, int(2.5 * SR)
    runs = [edit_region(source, SR, a, b, f0=ctx.f0, duration_factor=factor) for _ in range(4)]
    assert all(np.array_equal(runs[0], r) for r in runs[1:])
    assert len(runs[0]) == round((b - a) * factor)


def test_psola_edit_does_not_depend_on_call_history(source, ctx):
    a, b = 3 * SR, int(3.4 * SR)
    first = edit_region(source, SR, a, b, f0=ctx.f0, duration_factor=0.7)
    edit_region(source, SR, 10 * SR, 11 * SR, f0=ctx.f0, duration_factor=1.6)  # unrelated call
    assert np.array_equal(first, edit_region(source, SR, a, b, f0=ctx.f0, duration_factor=0.7))


def test_source_is_never_mutated(source, ctx):
    before = source.copy()
    edit_region(
        source, SR, SR, 2 * SR, f0=ctx.f0, duration_factor=0.6, pitch_edit=lambda t, f: f * 1.1
    )
    assert np.array_equal(before, source)


def test_gap_helpers_hit_exact_lengths_and_stay_finite(source):
    tone = RoomTone.from_audio(source, SR)
    rng = np.random.default_rng(0)
    gap = source[:6000].copy()
    longer = extend_gap(gap, 8000, tone, rng)
    shorter = shrink_gap(gap, 2000)
    assert len(longer) == len(gap) + 8000 and len(shorter) == 2000
    assert np.isfinite(longer).all() and np.isfinite(shorter).all()


def test_room_tone_matches_noise_floor_and_is_seeded(source):
    tone = RoomTone.from_audio(source, SR)
    a = tone.generate(SR, np.random.default_rng(1))
    b = tone.generate(SR, np.random.default_rng(1))
    assert np.array_equal(a, b)
    quiet = np.sqrt(np.mean(a**2))
    assert 1e-4 < quiet < 1e-2  # matches the ~-52 dB floor of the fixture, not digital silence
