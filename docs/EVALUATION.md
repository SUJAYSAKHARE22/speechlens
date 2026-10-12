# Evaluation report (development benchmark)

> **Read this first.** Every recording below is *synthetic*: three simulated speakers (espeak-ng voices sharing one text-driven prosody model) reading the Gettysburg Address, with flaws injected by our own engine. Detection thresholds and a few constants were tuned while looking at these same recordings. The numbers show that the machinery works end to end; they are **not** evidence of accuracy on human speech. Use `speechlens evaluate-human` on real, hand-annotated takes for that (see `docs/SOURCING.md`).

Setup: reference voice A; participants B and C (16 recordings each: 1 ideal, 5 ladder, 10 single-flaw). A region counts as detected when its time-IoU with a labelled flaw is >= 0.3.

## Headline results

| configuration | speaker | precision | recall | F1 | matched IoU | type acc. | start / end error (ms) |
|---|---|---|---|---|---|---|---|
| main (estimated alignment, leave-one-speaker-out calibration) | B | 1.0 | 0.821 | 0.902 | 0.91 | 1.0 | 12.0 / 18.1 |
| main (estimated alignment, leave-one-speaker-out calibration) | C | 0.875 | 0.75 | 0.808 | 0.838 | 1.0 | 13.7 / 12.6 |
| oracle alignment (ground-truth word timings) | B | 1.0 | 0.821 | 0.902 | 0.911 | 1.0 | 12.0 / 18.0 |
| oracle alignment (ground-truth word timings) | C | 0.875 | 0.75 | 0.808 | 0.841 | 1.0 | 13.3 / 12.6 |
| no calibration (default tolerances) | B | 1.0 | 0.821 | 0.902 | 0.91 | 1.0 | 12.0 / 18.1 |
| no calibration (default tolerances) | C | 0.88 | 0.786 | 0.83 | 0.832 | 0.955 | 15.4 / 12.8 |

Alignment effect (oracle minus estimated, F1): B +0.0, C +0.0, so alignment error is not the bottleneck on this data. Calibration effect (calibrated minus default, F1): B +0.0, C -0.022; it did not help on these voices (default tolerances were already wide enough) and slightly cost recall on C. Its purpose is robustness to more variable real speakers, which this benchmark cannot show.

## Scoring validity (main run)

| speaker | flawless take score | mean score by ladder level 0..5 | Spearman(level, score) | pairwise ordering | category attribution |
|---|---|---|---|---|---|
| B | 92.9 (false regions: 0.0) | 92.9 / 89.7 / 87.2 / 83.7 / 72.3 / 54.0 | -1.0 | 1.0 | 0.9 |
| C | 94.7 (false regions: 0.0) | 94.7 / 92.9 / 89.4 / 77.4 / 74.6 / 51.6 | -1.0 | 1.0 | 0.9 |

*Category attribution*: for single-flaw recordings, how often the rubric category matching the flaw loses the most points relative to the flawless take.

## Recall by flaw type (main run)

| flaw | B | C |
|---|---|---|
| drag | 0.8 | 0.667 |
| misplaced_pause | 1.0 | 1.0 |
| missing_pause | 1.0 | 0.5 |
| monotone | 1.0 | 1.0 |
| mumble | 0.5 | 0.5 |
| overlong_pause | 1.0 | 1.0 |
| rush | 0.6 | 0.75 |
| trail_off | 1.0 | 0.667 |
| uptalk | 0.5 | 0.333 |
| volume_collapse | 1.0 | 1.0 |

## Recall by injected severity (main run)

| severity | B | C |
|---|---|---|
| low (<0.4) | 0.0 | 0.0 |
| mid (0.4-0.7) | 0.786 | 0.733 |
| high (>=0.7) | 1.0 | 0.833 |

Low-severity flaws (barely perceptible by construction) are intentionally *not* flagged: they sit inside natural between-speaker variation. They still lower the continuous score slightly.

## Alignment accuracy (reference-DTW vs exact timings)

| speaker | boundaries | median | P90 | P99 | within 50 ms |
|---|---|---|---|---|---|
| B | 4544 | 6.9 ms | 18.6 ms | 37.3 ms | 99.4 % |
| C | 4544 | 9.6 ms | 29.0 ms | 60.4 ms | 98.7 % |

Simulated voices share one synthesis engine, so real human speakers will align less precisely.

## Threats to validity

1. **Circularity**: flaws were injected by our engine and detected by our analyser; both encode our idea of a flaw.
2. **Synthetic speakers**: espeak voices differ less from one another than real people do.
3. **Tuning on the test set**: `score_gain`, `missing_pause_ratio`, the terminal-pitch reliability gate and the silence threshold were chosen while inspecting these recordings. Only the tolerance calibration is leave-one-speaker-out.
4. **Tiny sample**: 2 speakers x 16 recordings (about 56 labelled flaws); recall for individual flaw types rests on 4-8 instances.
5. **No human judgements**: the rubric weights and the 0-100 scale are transparent heuristics, not calibrated to real adjudicators.
