"""Unit tests of the detector on hand-crafted deviation tables (no audio involved)."""

import numpy as np

from speechlens.analysis.compare import Deviations, z_scores
from speechlens.analysis.detect import detect_flaws
from speechlens.config import DetectionConfig, Tolerance
from speechlens.features import WordFeatures

N = 40


def _words(boundaries: dict[str, str] | dict[int, str]) -> list[WordFeatures]:
    def word(i: int) -> WordFeatures:
        return WordFeatures(
            index=i,
            text=f"w{i}",
            start=i * 0.4,
            end=i * 0.4 + 0.3,
            boundary=boundaries.get(i, "none"),
            n_syll=1,
            pause_before=0.1,
            pause_after=0.1,
            f0_mean_st=0.0,
            f0_std_st=2.0,
            f0_tail_st=-3.0,
            voiced_frac=0.8,
            energy_db=0.0,
            hf_db=-12.0,
            hnr_db=15.0,
        )

    return [word(i) for i in range(N)]


def _neutral() -> Deviations:
    return Deviations(
        mode="reference", tempo=1.0, pace=np.zeros(N), pause_obs=np.full(N, 0.1),
        pause_exp=np.full(N, 0.1), pitch=np.zeros(N), terminal=np.zeros(N), energy=np.zeros(N),
        clarity=np.zeros(N), detail={},
    )  # fmt: skip


def _detect(dev: Deviations, boundaries: dict[int, str] | None = None):
    words = _words(boundaries or {})
    z = z_scores(dev, Tolerance())
    return detect_flaws(dev, z, words, DetectionConfig())


def test_clean_speech_yields_nothing():
    assert _detect(_neutral()) == []


def test_rush_region_boundaries_are_word_exact():
    dev = _neutral()
    dev.pace[10:16] = -0.9  # 4.5 sigma
    regs = _detect(dev)
    assert [(r.type, r.i0, r.i1) for r in regs] == [("rush", 10, 15)]


def test_pauses_squeezed_by_a_rush_are_not_reported_twice():
    dev = _neutral()
    dev.pace[10:16] = -0.9
    dev.pause_exp[12], dev.pause_obs[12] = 0.5, 0.05  # a comma pause that the rush squeezed
    assert [r.type for r in _detect(dev, {12: "clause"})] == ["rush"]
    dev.pace[:] = 0.0  # the same squeezed pause *without* a rush is a real missing pause
    assert [r.type for r in _detect(dev, {12: "clause"})] == ["missing_pause"]


def test_long_pause_after_a_drag_is_consolidated_but_standalone_is_kept():
    dev = _neutral()
    dev.pace[20:25] = 0.9
    dev.pause_obs[22] = 1.5
    assert [r.type for r in _detect(dev, {22: "clause"})] == ["drag"]
    dev.pace[:] = 0.0
    assert [r.type for r in _detect(dev, {22: "clause"})] == ["overlong_pause"]
    assert [r.type for r in _detect(dev, {})] == ["misplaced_pause"]  # no punctuation there


def test_progressive_fade_at_a_sentence_end_is_trail_off_but_steady_loss_is_volume():
    dev = _neutral()
    dev.energy[25:30] = [0.0, -4.0, -8.0, -12.0, -18.0]  # decays toward the end of the sentence
    regs = _detect(dev, {29: "sentence"})
    assert [(r.type, r.i1) for r in regs] == [("trail_off", 29)]
    dev = _neutral()
    dev.energy[10:15] = -15.0  # constant drop mid-speech
    assert [r.type for r in _detect(dev, {29: "sentence"})] == ["volume_collapse"]


def test_terminal_rise_only_counts_at_sentence_ends():
    dev = _neutral()
    dev.terminal[12] = 6.0  # not a sentence end: ignored
    assert _detect(dev, {29: "sentence"}) == []
    dev.terminal[29] = 6.0
    assert [r.type for r in _detect(dev, {29: "sentence"})] == ["uptalk"]


def test_thresholds_are_respected():
    dev = _neutral()
    dev.pace[10:16] = -0.3  # 1.5 sigma: within natural variation
    dev.energy[20:25] = -5.0  # 1.7 sigma
    assert _detect(dev) == []
