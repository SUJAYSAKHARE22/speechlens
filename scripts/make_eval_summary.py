"""Build docs/EVALUATION.md from the JSON reports in docs/evaluation (no hand-copied numbers)."""

import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "docs" / "evaluation"
load = lambda n: json.loads((D / f"{n}.json").read_text())  # noqa: E731
RUNS = [("main (estimated alignment, leave-one-speaker-out calibration)", "dev_synthetic"),
        ("oracle alignment (ground-truth word timings)", "dev_synthetic_{}_oracle"),
        ("no calibration (default tolerances)", "dev_synthetic_uncalibrated")]  # fmt: skip


def name(tag: str, spk: str) -> str:
    return tag.format(spk) if "{}" in tag else f"{tag}_{spk}"


L = ["# Evaluation report (development benchmark)", "",
     "> **Read this first.** Every recording below is *synthetic*: three simulated speakers (espeak-ng "
     "voices sharing one text-driven prosody model) reading the Gettysburg Address, with flaws "
     "injected by our own engine. Detection thresholds and a few constants were tuned while looking "
     "at these same recordings. The numbers show that the machinery works end to end; they are "
     "**not** evidence of accuracy on human speech. Use `speechlens evaluate-human` on real, "
     "hand-annotated takes for that (see `docs/SOURCING.md`).", "",
     "Setup: reference voice A; participants B and C (16 recordings each: 1 ideal, 5 ladder, 10 "
     "single-flaw). A region counts as detected when its time-IoU with a labelled flaw is >= 0.3.", ""]  # fmt: skip
L += ["## Headline results", "", "| configuration | speaker | precision | recall | F1 | matched IoU | type acc. | start / end error (ms) |",
      "|---|---|---|---|---|---|---|---|"]  # fmt: skip
for title, tag in RUNS:
    for spk in "BC":
        r = load(name(tag, spk))
        d = r["detection"]["iou>=0.3"]
        L.append(f"| {title} | {spk} | {d['precision']} | {d['recall']} | {d['f1']} | {r['matched_mean_iou']} | "
                 f"{r['type_accuracy_among_matched']} | {r['median_start_error_ms']} / {r['median_end_error_ms']} |")  # fmt: skip
f1 = lambda tag, spk: load(name(tag, spk))["detection"]["iou>=0.3"]["f1"]  # noqa: E731
cal = {
    spk: round(f1("dev_synthetic", spk) - f1("dev_synthetic_uncalibrated", spk), 3) for spk in "BC"
}
ora = {spk: round(f1("dev_synthetic_{}_oracle", spk) - f1("dev_synthetic", spk), 3) for spk in "BC"}
L += ["", f"Alignment effect (oracle minus estimated, F1): B {ora['B']:+}, C {ora['C']:+}, so alignment error is "
      f"not the bottleneck on this data. Calibration effect (calibrated minus default, F1): B {cal['B']:+}, "
      f"C {cal['C']:+}; it did not help on these voices (default tolerances were already wide enough) and "
      "slightly cost recall on C. Its purpose is robustness to more variable real speakers, which this "
      "benchmark cannot show.", ""]  # fmt: skip
L += ["## Scoring validity (main run)", "", "| speaker | flawless take score | mean score by ladder level 0..5 | Spearman(level, score) | pairwise ordering | category attribution |", "|---|---|---|---|---|---|"]  # fmt: skip
for spk in "BC":
    r = load(name("dev_synthetic", spk))
    s = r["score_validation"]
    lv = " / ".join(str(s["mean_score_by_level"][str(k)]) for k in range(6))
    L.append(f"| {spk} | {r['ideal_recordings']['mean_score']} (false regions: {r['ideal_recordings']['false_regions_per_recording']}) | {lv} | {s['spearman_level_vs_score']} | {s['pairwise_ordering_accuracy']} | {s['category_attribution_accuracy']} |")  # fmt: skip
L += ["", "*Category attribution*: for single-flaw recordings, how often the rubric category matching the "
      "flaw loses the most points relative to the flawless take.", ""]  # fmt: skip
types = sorted({t for spk in "BC" for t in load(name("dev_synthetic", spk))["recall_by_type"]})
L += ["## Recall by flaw type (main run)", "", "| flaw | B | C |", "|---|---|---|"]
for t in types:
    L.append(
        f"| {t} | {load('dev_synthetic_B')['recall_by_type'].get(t, '-')} | {load('dev_synthetic_C')['recall_by_type'].get(t, '-')} |"
    )
bins = ["low (<0.4)", "mid (0.4-0.7)", "high (>=0.7)"]
L += ["", "## Recall by injected severity (main run)", "", "| severity | B | C |", "|---|---|---|"]
for b in bins:
    L.append(
        f"| {b} | {load('dev_synthetic_B')['recall_by_severity'].get(b, '-')} | {load('dev_synthetic_C')['recall_by_severity'].get(b, '-')} |"
    )
L += ["", "Low-severity flaws (barely perceptible by construction) are intentionally *not* flagged: they sit "
      "inside natural between-speaker variation. They still lower the continuous score slightly.", ""]  # fmt: skip
a = json.loads((D / "alignment_accuracy.json").read_text())
L += ["## Alignment accuracy (reference-DTW vs exact timings)", "", "| speaker | boundaries | median | P90 | P99 | within 50 ms |", "|---|---|---|---|---|---|"]  # fmt: skip
for spk in "BC":
    x = a[spk]
    L.append(
        f"| {spk} | {x['n_boundaries']} | {x['median_ms']} ms | {x['p90_ms']} ms | {x['p99_ms']} ms | {x['within_50ms_pct']} % |"
    )
L += ["", "Simulated voices share one synthesis engine, so real human speakers will align less precisely.", "",
      "## Threats to validity", "",
      "1. **Circularity**: flaws were injected by our engine and detected by our analyser; both encode our idea of a flaw.",
      "2. **Synthetic speakers**: espeak voices differ less from one another than real people do.",
      "3. **Tuning on the test set**: `score_gain`, `missing_pause_ratio`, the terminal-pitch reliability gate and "
      "the silence threshold were chosen while inspecting these recordings. Only the tolerance calibration is leave-one-speaker-out.",
      "4. **Tiny sample**: 2 speakers x 16 recordings (about 56 labelled flaws); recall for individual flaw types rests on 4-8 instances.",
      "5. **No human judgements**: the rubric weights and the 0-100 scale are transparent heuristics, not calibrated to real adjudicators.", ""]  # fmt: skip
(D.parent / "EVALUATION.md").write_text("\n".join(L), encoding="utf-8")
print("wrote docs/EVALUATION.md")
