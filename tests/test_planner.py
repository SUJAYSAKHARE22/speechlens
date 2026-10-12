import numpy as np
import pytest

from speechlens.dataset.flaws import FLAWS, FlawType
from speechlens.dataset.planner import RegionPlanner
from speechlens.text import boundary_kind


@pytest.mark.parametrize("ftype", list(FlawType))
def test_every_flaw_type_can_be_planned_and_respects_anchor(alignment, ftype):
    words = alignment.words
    req = RegionPlanner(words, np.random.default_rng(3)).plan(ftype, 0.5)
    assert req is not None
    lo, hi = FLAWS[ftype].span
    assert lo <= req.word_end - req.word_start + 1 <= hi
    anchor = FLAWS[ftype].anchor
    kinds = [boundary_kind(w.punct_after) for w in words]
    if anchor == "sentence_end":
        assert kinds[req.word_end] == "sentence"
    if anchor == "boundary_none":
        assert kinds[req.word_start] == "none"
    if anchor == "boundary_punct":
        assert kinds[req.word_start] != "none"
    if FLAWS[ftype].min_gap_s:
        gap = words[req.word_start + 1].start - words[req.word_start].end
        assert gap >= FLAWS[ftype].min_gap_s


def test_planned_regions_never_overlap_and_keep_a_guard_word(alignment):
    rng = np.random.default_rng(11)
    planner = RegionPlanner(alignment.words, rng)
    regions = []
    for ftype in list(FlawType) * 2:
        r = planner.plan(ftype, 0.5)
        if r:
            regions.append((r.word_start, r.word_end))
    regions.sort()
    assert len(regions) >= 6
    for (_, e1), (s2, _) in zip(regions, regions[1:], strict=False):
        assert s2 - e1 >= 2  # at least one untouched word between regions


def test_planner_is_deterministic(alignment):
    a = RegionPlanner(alignment.words, np.random.default_rng(5)).plan(FlawType.RUSH, 0.4)
    b = RegionPlanner(alignment.words, np.random.default_rng(5)).plan(FlawType.RUSH, 0.4)
    assert a == b


def test_uptalk_is_only_planned_on_voiced_words(alignment, voiced):
    for seed in range(12):
        req = RegionPlanner(alignment.words, np.random.default_rng(seed), voiced).plan(
            FlawType.UPTALK, 0.7
        )
        assert req is not None and voiced[req.word_end] >= 0.6
