import numpy as np
import pytest

from speechlens.analysis.calibration import calibrate_tolerance
from speechlens.analysis.compare import Deviations
from speechlens.config import Tolerance
from speechlens.evaluation.metrics import iou, match_regions, median_ms, prf
from speechlens.evaluation.run import evaluate


def test_iou_basics():
    assert iou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    assert iou((0, 1), (2, 3)) == 0.0 and iou((0, 1), (0, 1)) == 1.0


def test_matching_is_one_to_one_and_prefers_best_overlap():
    pred = [("a", 0.0, 2.0), ("b", 0.2, 1.9), ("c", 5.0, 6.0)]
    gt = [("a", 0.0, 2.0), ("x", 8.0, 9.0)]
    ms = match_regions(pred, gt, 0.3)
    assert [(m.pred, m.gt) for m in ms] == [(0, 0)]  # second prediction cannot reuse the same GT


def test_prf_and_median():
    assert prf(8, 2, 2) == {"precision": 0.8, "recall": 0.8, "f1": 0.8}
    assert prf(0, 0, 0)["f1"] == 0.0
    assert median_ms([0.01, 0.03, 0.02]) == 20.0 and median_ms([]) is None


def _dev(scale: float) -> Deviations:
    n = 50
    rng = np.random.default_rng(0)
    z = lambda s: rng.normal(0, s, n)  # noqa: E731
    return Deviations(
        mode="reference",
        tempo=1.0,
        pace=z(0.1 * scale),
        pause_obs=np.full(n, 0.3 + 0.5 * scale),
        pause_exp=np.full(n, 0.3),
        pitch=z(0.5 * scale),
        terminal=z(5 * scale),
        energy=z(8 * scale),
        clarity=z(1 * scale),
        detail={},
    )


def test_calibration_only_widens_and_is_noop_for_tame_data():
    base = Tolerance()
    assert calibrate_tolerance([], base) == base
    tame = calibrate_tolerance([_dev(0.01)], base)
    assert tame == base  # tiny natural variation can never make the detector more sensitive
    wide = calibrate_tolerance([_dev(1.0)], base)
    assert wide.energy_sigma > base.energy_sigma and wide.terminal_sigma > base.terminal_sigma
    assert wide.pitch_sigma >= base.pitch_sigma


def test_evaluate_reports_sane_metrics(reference, b_dataset):
    from speechlens.analysis.pipeline import Analyzer

    parts = [p for p in b_dataset.values()]
    rep = evaluate(Analyzer(), reference, parts)
    d = rep["detection"]["iou>=0.3"]
    assert d["precision"] >= 0.6 and d["recall"] >= 0.6
    assert rep["matched_mean_iou"] > 0.7 and rep["type_accuracy_among_matched"] > 0.9
    assert rep["ideal_recordings"]["false_regions_per_recording"] == 0.0
    assert rep["score_validation"]["spearman_level_vs_score"] < -0.8
    assert rep["median_start_error_ms"] < 60
    assert set(rep["recall_by_type"]) <= {
        "rush",
        "drag",
        "misplaced_pause",
        "missing_pause",
        "overlong_pause",
        "monotone",
        "trail_off",
        "volume_collapse",
        "mumble",
        "uptalk",
    }


def test_human_annotation_loader_and_evaluation(tmp_path, reference, b_dataset):
    import json

    from speechlens.analysis.pipeline import Analyzer
    from speechlens.audio import save_audio
    from speechlens.evaluation.run import load_human_annotations

    flawed = b_dataset["S_overlong_pause_60"]
    save_audio(tmp_path / "take.wav", flawed.audio, 16_000)
    save_audio(tmp_path / "good.wav", b_dataset["L0_ideal"].audio, 16_000)
    t, s, e, _ = flawed.gt[0]
    doc = {"recordings": [{"audio": "take.wav", "flaws": [{"type": t, "start": s, "end": e}]},
                          {"audio": "good.wav", "flaws": []}]}  # fmt: skip
    (tmp_path / "ann.json").write_text(json.dumps(doc), encoding="utf-8")
    parts = load_human_annotations(tmp_path / "ann.json", reference)
    assert [p.kind for p in parts] == ["human", "ideal"]
    rep = evaluate(Analyzer(), reference, parts)
    assert rep["detection"]["iou>=0.3"]["recall"] == 1.0
    assert rep["ideal_recordings"]["false_regions_per_recording"] == 0.0

    bad = {
        "recordings": [{"audio": "take.wav", "flaws": [{"type": "nonsense", "start": 1, "end": 2}]}]
    }
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_human_annotations(tmp_path / "bad.json", reference)
