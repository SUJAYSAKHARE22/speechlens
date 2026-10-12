import json

import numpy as np
import pytest

from speechlens.analysis.models import AnalysisResult
from speechlens.analysis.pipeline import Analyzer
from speechlens.analysis.rubric import PRESETS, score_speech
from speechlens.config import Config, DetectionConfig
from speechlens.evaluation.metrics import iou
from tests.conftest import SR


@pytest.fixture(scope="module")
def analyzer():
    return Analyzer(Config())


@pytest.fixture(scope="module")
def results(analyzer, reference, b_dataset):
    """Analyse every speaker-B variant once against the reference voice (a different speaker)."""
    return {
        vid: analyzer.analyze(p.audio, p.transcript, reference=reference)
        for vid, p in b_dataset.items()
    }


def test_reference_compared_with_itself_is_perfect(analyzer, reference):
    r = analyzer.analyze(
        reference.audio, reference.transcript, reference=reference, alignment=reference.alignment
    )
    assert r.scores.overall == 100.0 and r.regions == []


def test_a_good_reading_by_another_speaker_is_not_flagged(results):
    r = results["L0_ideal"]
    assert r.regions == [] and r.scores.overall >= 85
    assert 0.85 < r.tempo_factor < 1.2  # B is a little slower than A, and that is normalised away


def test_analysis_is_invariant_to_recording_gain(analyzer, reference, b_dataset):
    p = b_dataset["L3_mediocre"]
    a = analyzer.analyze(p.audio, p.transcript, reference=reference)
    b = analyzer.analyze(p.audio * 0.2, p.transcript, reference=reference)
    assert [(g.type, g.word_start, g.word_end) for g in a.regions] == [
        (g.type, g.word_start, g.word_end) for g in b.regions
    ]
    assert abs(a.scores.overall - b.scores.overall) < 1.0


@pytest.mark.parametrize(
    "variant",
    [
        "S_overlong_pause_60",
        "S_misplaced_pause_60",
        "S_volume_collapse_60",
        "S_monotone_60",
        "S_trail_off_60",
        "S_missing_pause_60",
    ],
)
def test_isolated_flaws_are_grounded_with_correct_type(results, b_dataset, variant):
    gt = b_dataset[variant].gt[0]
    best = max(results[variant].regions, key=lambda g: iou((g.start, g.end), (gt[1], gt[2])))
    assert iou((best.start, best.end), (gt[1], gt[2])) >= 0.5
    assert best.type == gt[0]


@pytest.mark.parametrize("variant", ["S_rush_60", "S_drag_60"])
def test_pacing_flaws_are_grounded_and_typed(results, b_dataset, variant):
    gt = b_dataset[variant].gt[0]
    hits = [g for g in results[variant].regions if iou((g.start, g.end), (gt[1], gt[2])) >= 0.3]
    assert hits and hits[0].type == gt[0]


def test_every_region_carries_a_numeric_causal_explanation(results):
    regions = [g for r in results.values() for g in r.regions]
    assert len(regions) >= 10
    for g in regions:
        assert g.words and g.start < g.end and 0.2 <= g.severity <= 1 and 0.5 <= g.confidence <= 1
        assert g.headline and g.explanation and g.suggestion
        assert any(ch.isdigit() for ch in g.rationale)  # the maths is spelled out
        assert "sigma" in g.rationale or "std" in g.rationale or "flagged" in g.rationale
        if g.type != "missing_pause":  # that one is flagged by the ratio rule, not by |z|
            assert abs(g.evidence.z) >= 2.0
        assert g.family in {"pace", "pause", "pitch", "energy", "clarity"}


def test_pause_regions_expose_the_pause_interval(results):
    g = next(g for g in results["S_overlong_pause_60"].regions if g.type == "overlong_pause")
    assert g.focus_start is not None and g.start <= g.focus_start < g.focus_end <= g.end
    assert g.focus_end - g.focus_start == pytest.approx(g.evidence.observed, abs=0.1)


def test_scores_fall_down_the_quality_ladder(results):
    ladder = [(int(v[1]), r.scores.overall) for v, r in results.items() if v.startswith("L")]
    by_level = dict(ladder)
    assert by_level[0] > by_level[5] + 20
    assert by_level[0] >= by_level[3] >= by_level[5]
    from scipy.stats import spearmanr

    assert spearmanr(*zip(*ladder, strict=True)).statistic < -0.8


def test_single_flaw_lowers_its_own_category_most(results):
    base = results["L0_ideal"].scores.categories
    for variant, cat in [
        ("S_volume_collapse_60", "energy"),
        ("S_monotone_60", "pitch"),
        ("S_overlong_pause_60", "pauses"),
        ("S_rush_60", "pacing"),
    ]:
        drops = {c: base[c] - v for c, v in results[variant].scores.categories.items()}
        assert max(drops, key=drops.get) == cat, variant


def test_norm_mode_runs_without_a_reference_and_says_so(analyzer, b_dataset):
    p = b_dataset["L5_botched"]
    r = analyzer.analyze(p.audio, p.transcript, alignment=p.alignment)
    assert r.mode == "norm" and r.tempo_factor is None and r.series.f0_st_ref is None
    assert any("No reference" in w for w in r.warnings)
    ideal = analyzer.analyze(
        b_dataset["L0_ideal"].audio, p.transcript, alignment=b_dataset["L0_ideal"].alignment
    )
    assert ideal.scores.overall > r.scores.overall


def test_result_is_strict_json_with_aligned_series(results):
    r = results["L5_botched"]
    text = r.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    assert AnalysisResult.model_validate_json(text) == r
    s = r.series
    n = len(s.t)
    assert len(s.f0_st) == len(s.f0_st_ref) == len(s.energy_db) == len(s.energy_db_ref) == n
    assert s.t == sorted(s.t) and len(r.words) == len(results["L0_ideal"].words)
    json.loads(text)


def test_transcript_mismatch_and_unknown_preset_are_rejected(analyzer, reference, b_dataset):
    p = b_dataset["L0_ideal"]
    with pytest.raises(ValueError, match="word count"):
        analyzer.analyze(p.audio, "far too short", reference=reference)
    with pytest.raises(ValueError, match="unknown preset"):
        analyzer.analyze(p.audio, p.transcript, reference=reference, preset="nope")


def test_rubric_presets_are_valid_and_change_the_overall_score():
    assert all(abs(sum(w.values()) - 1.0) < 1e-9 for w in PRESETS.values())
    z = {k: np.zeros(20) for k in ["pace", "pause", "pitch", "terminal", "energy", "clarity"]}
    z["energy"][5:15] = -6.0  # a long volume problem
    cfg = DetectionConfig()
    a, b = score_speech(z, "declamation", cfg), score_speech(z, "extemporaneous", cfg)
    assert a.categories["energy"] == b.categories["energy"] < 60
    assert a.overall < b.overall  # declamation weights energy more heavily
    assert score_speech({k: np.zeros(20) for k in z}, "balanced", cfg).overall == 100.0


def test_misaligned_transcript_audio_raises_a_warning(analyzer, reference):
    noise = np.random.default_rng(1).standard_normal(len(reference.audio)) * 0.05
    r = analyzer.analyze(noise, reference.transcript, reference=reference)
    assert any("alignment cost" in w.lower() for w in r.warnings)


def test_sample_rate_constant():
    assert SR == 16_000


def test_displayed_math_is_self_consistent(results):
    """z == delta / sigma for every region, and the rationale quotes those exact numbers."""
    regions = [g for r in results.values() for g in r.regions]
    assert len(regions) >= 10
    for g in regions:
        ev = g.evidence
        assert ev.sigma > 0 and ev.z == pytest.approx(ev.delta / ev.sigma, abs=1e-6), g.headline
        assert f"{ev.z:+.1f}" in g.rationale, (g.type, g.rationale)
        assert ev.delta_unit in {"octaves", "s", "semitones", "dB"}


def test_quiet_passages_do_not_create_phantom_pacing_flaws(results, b_dataset):
    """A volume drop must be reported as a volume drop only (regression: clipped quiet words)."""
    types = [g.type for g in results["S_volume_collapse_60"].regions]
    assert types == ["volume_collapse"], types


def test_a_single_misplaced_pause_visibly_lowers_the_pause_score(results):
    ideal = results["L0_ideal"].scores.categories["pauses"]
    flawed = results["S_misplaced_pause_60"].scores.categories["pauses"]
    assert ideal - flawed > 15
