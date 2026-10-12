"""Run the analyzer over labelled participants and aggregate the evaluation metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from speechlens.analysis.detect import FAMILY
from speechlens.analysis.pipeline import Analyzer, Reference
from speechlens.audio import load_audio
from speechlens.evaluation.metrics import Region, match_regions, median_ms, prf
from speechlens.schema import Alignment, VariantLabel

CATEGORY_OF_FAMILY = {
    "pace": "pacing", "pause": "pauses", "pitch": "pitch", "energy": "energy", "clarity": "clarity",
}  # fmt: skip
IOU_THRESHOLDS = (0.3, 0.5)


@dataclass
class Participant:
    id: str
    audio: np.ndarray
    transcript: str
    gt: list[tuple[str, float, float, float]]  # (type, start, end, severity)
    kind: str
    level: int | None
    alignment: Alignment | None = None  # oracle alignment (optional)


def load_dataset_participants(
    dataset_root: str | Path, base_id: str, sr: int = 16_000
) -> list[Participant]:
    root = Path(dataset_root)
    transcript = (root / base_id / "transcript.txt").read_text(encoding="utf-8").strip()
    out = []
    for lp in sorted((root / base_id / "labels").glob("*.json")):
        lb = VariantLabel.load(lp)
        out.append(
            Participant(
                id=f"{lb.base_id}/{lb.variant_id}", audio=load_audio(root / lb.audio_file, sr),
                transcript=transcript, kind=lb.kind, level=lb.level, alignment=lb.alignment,
                gt=[(f.type, f.start, f.end, f.severity) for f in lb.flaws],
            )
        )  # fmt: skip
    return out


@dataclass
class _Acc:
    tp: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    fp: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    fn: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    type_ok: int = 0
    family_ok: int = 0
    matched: int = 0
    ious: list[float] = field(default_factory=list)
    start_err: list[float] = field(default_factory=list)
    end_err: list[float] = field(default_factory=list)
    by_type: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    by_sev: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))


def _sev_bin(s: float) -> str:
    return "low (<0.4)" if s < 0.4 else "mid (0.4-0.7)" if s < 0.7 else "high (>=0.7)"


def evaluate(
    analyzer: Analyzer, reference: Reference, participants: list[Participant],
    oracle_alignment: bool = False, preset: str = "balanced",
) -> dict:  # fmt: skip
    acc = _Acc()
    ideal_fp, ideal_scores, ideal_n = 0, [], 0
    ladder: list[tuple[int, float]] = []
    attribution: list[bool] = []
    per_recording = []

    ideal_cats: dict[str, float] | None = None
    results = {}
    for p in participants:
        r = analyzer.analyze(
            p.audio, p.transcript, reference=reference, preset=preset,
            alignment=p.alignment if oracle_alignment else None,
        )  # fmt: skip
        results[p.id] = r
        if p.kind == "ideal":
            ideal_cats = r.scores.categories
    for p in participants:
        r = results[p.id]
        pred: list[Region] = [(g.type, g.start, g.end) for g in r.regions]
        gt: list[Region] = [(t, s, e) for t, s, e, _ in p.gt]
        per_recording.append(
            {"id": p.id, "level": p.level, "kind": p.kind, "overall": r.scores.overall,
             "n_gt": len(gt), "n_pred": len(pred), "pred_types": [x[0] for x in pred]}
        )  # fmt: skip
        if p.level is not None:
            ladder.append((p.level, r.scores.overall))
        if not p.gt:
            ideal_fp += len(pred)
            ideal_n += 1
            ideal_scores.append(r.scores.overall)
            continue
        for thr in IOU_THRESHOLDS:
            ms = match_regions(pred, gt, thr)
            acc.tp[thr] += len(ms)
            acc.fp[thr] += len(pred) - len(ms)
            acc.fn[thr] += len(gt) - len(ms)
        for m in match_regions(pred, gt, IOU_THRESHOLDS[0]):
            acc.matched += 1
            acc.ious.append(m.iou)
            acc.type_ok += pred[m.pred][0] == gt[m.gt][0]
            acc.family_ok += FAMILY[pred[m.pred][0]] == FAMILY[gt[m.gt][0]]
            acc.start_err.append(abs(pred[m.pred][1] - gt[m.gt][1]))
            acc.end_err.append(abs(pred[m.pred][2] - gt[m.gt][2]))
        hit = {m.gt for m in match_regions(pred, gt, IOU_THRESHOLDS[0])}
        for j, (t, _, _, sev) in enumerate(p.gt):
            acc.by_type[t][1] += 1
            acc.by_type[t][0] += j in hit
            acc.by_sev[_sev_bin(sev)][1] += 1
            acc.by_sev[_sev_bin(sev)][0] += j in hit
        if p.kind == "single" and ideal_cats is not None:
            fam = CATEGORY_OF_FAMILY[FAMILY[p.gt[0][0]]]
            drops = {c: ideal_cats[c] - v for c, v in r.scores.categories.items()}
            attribution.append(max(drops, key=drops.get) == fam)

    levels = sorted({lv for lv, _ in ladder})
    mean_by_level = {
        lv: round(float(np.mean([s for l2, s in ladder if l2 == lv])), 1) for lv in levels
    }
    rho = (
        float(spearmanr([lv for lv, _ in ladder], [s for _, s in ladder]).statistic)
        if len(ladder) > 2
        else None
    )
    pairs = [(a, b) for a in ladder for b in ladder if a[0] < b[0]]
    order_acc = float(np.mean([a[1] > b[1] for a, b in pairs])) if pairs else None
    return {
        "n_recordings": len(participants),
        "detection": {f"iou>={t}": prf(acc.tp[t], acc.fp[t], acc.fn[t]) for t in IOU_THRESHOLDS},
        "matched_mean_iou": round(float(np.mean(acc.ious)), 3) if acc.ious else None,
        "type_accuracy_among_matched": round(acc.type_ok / acc.matched, 3) if acc.matched else None,
        "family_accuracy_among_matched": round(acc.family_ok / acc.matched, 3)
        if acc.matched
        else None,
        "median_start_error_ms": median_ms(acc.start_err),
        "median_end_error_ms": median_ms(acc.end_err),
        "recall_by_type": {t: round(h / n, 3) for t, (h, n) in sorted(acc.by_type.items())},
        "recall_by_severity": {k: round(h / n, 3) for k, (h, n) in sorted(acc.by_sev.items())},
        "ideal_recordings": {
            "n": ideal_n,
            "false_regions_per_recording": round(ideal_fp / ideal_n, 2) if ideal_n else None,
            "mean_score": round(float(np.mean(ideal_scores)), 1) if ideal_scores else None,
        },  # fmt: skip
        "score_validation": {
            "mean_score_by_level": mean_by_level,
            "spearman_level_vs_score": None if rho is None else round(rho, 3),
            "pairwise_ordering_accuracy": None if order_acc is None else round(order_acc, 3),
            "category_attribution_accuracy": round(float(np.mean(attribution)), 3)
            if attribution
            else None,
        },  # fmt: skip
        "per_recording": per_recording,
    }


def to_markdown(title: str, report: dict) -> str:
    d, s = report["detection"], report["score_validation"]
    lines = [f"### {title}", "", f"Recordings: {report['n_recordings']}", "",
             "| IoU threshold | precision | recall | F1 |", "|---|---|---|---|"]  # fmt: skip
    lines += [f"| {k} | {v['precision']} | {v['recall']} | {v['f1']} |" for k, v in d.items()]
    lines += ["", f"- matched mean IoU: **{report['matched_mean_iou']}**; median start / end error: "
              f"**{report['median_start_error_ms']} / {report['median_end_error_ms']} ms**",
              f"- type accuracy among matched: **{report['type_accuracy_among_matched']}** "
              f"(family: {report['family_accuracy_among_matched']})",
              f"- ideal (flawless) recordings: {report['ideal_recordings']}",
              f"- score vs ladder level: Spearman **{s['spearman_level_vs_score']}**, pairwise ordering "
              f"**{s['pairwise_ordering_accuracy']}**, category attribution **{s['category_attribution_accuracy']}**",
              f"- mean score by level: {s['mean_score_by_level']}", "",
              "| flaw type | recall |", "|---|---|"]  # fmt: skip
    lines += [f"| {t} | {r} |" for t, r in report["recall_by_type"].items()]
    lines += ["", "| severity | recall |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in report["recall_by_severity"].items()]
    return "\n".join(lines) + "\n"


def save_report(report: dict, out: str | Path, title: str) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text(to_markdown(title, report), encoding="utf-8")


def evaluate_corpus(
    corpus: str | Path, reference_id: str, speakers: list[str], config=None,
    calibrate: bool = True, oracle_alignment: bool = False,
) -> dict[str, dict]:  # fmt: skip
    """Evaluate every simulated speaker's dataset against ``reference_id``'s ideal recording.

    With ``calibrate`` the tolerance for each speaker is learned from the *other* speakers' good
    readings only (leave-one-speaker-out), so no evaluated recording influences its own thresholds.
    """
    from speechlens.analysis.calibration import calibrate_tolerance
    from speechlens.analysis.compare import compare_to_reference
    from speechlens.config import Config
    from speechlens.features import extract_speech_features

    cfg = config or Config()
    corpus = Path(corpus)
    ref = Reference.load(corpus / "bases" / reference_id)
    parts = {s: load_dataset_participants(corpus / "datasets" / s, s) for s in speakers}

    def good(s: str):
        p = next(q for q in parts[s] if q.kind == "ideal")
        par = extract_speech_features(p.audio, 16_000, ref.aligner.align(p.audio))
        return compare_to_reference(ref.features, par, cfg.analysis.detection.pitch_half_window)

    good_cache = {s: good(s) for s in speakers} if calibrate else {}
    out = {}
    for s in speakers:
        c = cfg.model_copy(deep=True)
        if calibrate and len(speakers) > 1:
            c.analysis.tolerance = calibrate_tolerance(
                [d for k, d in good_cache.items() if k != s], cfg.analysis.tolerance
            )
        out[s] = evaluate(Analyzer(c), ref, parts[s], oracle_alignment=oracle_alignment)
        out[s]["tolerance_used"] = c.analysis.tolerance.model_dump()
    return out


def load_human_annotations(
    path: str | Path, reference: Reference, sr: int = 16_000
) -> list[Participant]:
    """Load human-recorded takes with hand-annotated flaw spans (the non-circular test set).

    JSON format (audio paths are relative to the annotation file)::

        {"recordings": [
            {"audio": "take1.wav", "flaws": [{"type": "rush", "start": 12.1, "end": 14.0}]},
            {"audio": "good_take.wav", "flaws": []}
        ]}

    A recording with no flaws is treated as a flawless take (used to count false alarms).
    """
    from speechlens.dataset.flaws import FlawType

    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for rec in doc["recordings"]:
        flaws = []
        for f in rec.get("flaws", []):
            FlawType(f["type"])  # validates the label vocabulary
            if not 0 <= f["start"] < f["end"]:
                raise ValueError(f"bad flaw span in {rec['audio']}: {f}")
            flaws.append(
                (f["type"], float(f["start"]), float(f["end"]), float(f.get("severity", 0.5)))
            )
        out.append(
            Participant(
                id=rec["audio"], audio=load_audio(path.parent / rec["audio"], sr),
                transcript=reference.transcript, gt=flaws, kind="human" if flaws else "ideal",
                level=None,
            )
        )  # fmt: skip
    return out
