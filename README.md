# SpeechLens

**Contrastive speech analytics with temporal flaw grounding.** Compare a speaker's delivery with a
good reference reading of the *same text*, find the exact stretches where delivery goes wrong, and
explain each one with the acoustic evidence and the maths behind it.

*Multimodal AI Hackathon 2026, Track C.*

* **Dataset engine**: injects ten delivery flaws at controllable severity into a baseline recording,
  producing a quality ladder with **exact, measured ground-truth timestamps** (bit-reproducible).
* **Analysis**: forced alignment, speaker-normalised features (F0, energy, pauses, rate, clarity),
  word-by-word comparison, flaw regions with word-boundary start/end, numeric causal explanations.
* **Dashboard**: upload audio and text; time-series overlay of baseline vs participant, shaded flaw
  regions, click-to-seek notes with a "show the math" panel, five-category rubric score.

> **Honest status.** Evidence so far is on *synthetic* speakers (see `docs/EVALUATION.md`):
> precision 1.0/0.88, recall 0.82/0.75, matched IoU 0.91/0.84, boundary error 12-18 ms, zero false
> regions on flawless takes by other voices. Thresholds were tuned on those recordings, so treat
> this as a working-pipeline benchmark, not proof of accuracy on human speech. The repo includes the
> tooling to measure that (`speechlens evaluate-human`, `docs/SOURCING.md`).

## Quick start

```bash
sudo apt install espeak-ng libsndfile1 ffmpeg     # macOS: brew install espeak-ng libsndfile ffmpeg
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"

speechlens serve                  # dashboard at http://127.0.0.1:8000 (uses data/demo)
pytest                            # 118 tests (about 10 minutes on one core)
```
Docker: `docker build -t speechlens . && docker run --rm -p 8000:8000 speechlens`.
`espeak-ng` is needed only for the simulated speakers and reference-free alignment; the dashboard
with a chosen reference works without it.

## Reproduce the dataset and evaluation

```bash
speechlens make-dev-corpus --out data/dev_corpus   # 3 simulated speakers, 48 labelled recordings
speechlens verify data/dev_corpus/datasets/B       # hashes, intervals, alignment integrity
speechlens evaluate                                # leave-one-speaker-out calibration (about 2.5 min)
speechlens evaluate --oracle                       # with ground-truth alignments
python scripts/make_eval_summary.py                # rebuilds docs/EVALUATION.md from the JSON
```

## Use your own recordings

```bash
# 1. data/bases/<id>/{audio.wav, transcript.txt, alignment.json}   (see docs/SOURCING.md)
speechlens build-dataset --base data/bases --out data/dataset      # ideal + ladder + single flaws
speechlens analyze take.wav --reference data/bases/<id> --out result.json
speechlens calibrate --reference data/bases/<id> --good good1.wav good2.wav --out tolerance.yaml
speechlens evaluate-human --reference data/bases/<id> --annotations annotations.json
```

| command | purpose |
|---|---|
| `make-dev-corpus`, `make-fixtures` | simulated speakers for offline development |
| `build-dataset`, `verify` | build and integrity-check the contrastive dataset |
| `analyze` | analyse one recording (reference or reference-free) |
| `calibrate` | learn natural between-speaker tolerance from good readings |
| `evaluate`, `evaluate-human` | metrics against injected ground truth / human annotations |
| `make-demo`, `serve` | assemble demo data; run the dashboard |

All settings live in one validated file: `configs/default.yaml` (`--config` to override).

## Repository map

```
src/speechlens/
  dataset/     injection engine (flaws, timeline, planner, DSP), builder, verifier, fixtures
  alignment/   reference-DTW, TTS-DTW, energy boundary refinement, MFA TextGrid import
  features/    frame and word features, speaker normalisation
  analysis/    compare, detect, explain, rubric, calibration, pipeline
  evaluation/  IoU matching, metrics, corpus and human-annotation runners
  api/         FastAPI app, demo builder, single-file dashboard (Plotly vendored, no CDN)
docs/          TECHNICAL.md (<= 6 pages), DATASET_CARD.md, EVALUATION.md, SOURCING.md, VIDEO_SCRIPT.md
data/demo/     reference + 7 sample recordings with ground truth (12 MB)
tests/         118 tests;  scripts/  report generator;  Dockerfile, Makefile, CI workflow
```

## Hackathon deliverables

| deliverable | where |
|---|---|
| Code and documentation | this repository |
| Custom dataset (paired good/bad spectrum, transcripts, alignment labels) | `data/demo/` (sample) and reproducible via `make-dev-corpus`; real-speech data: your baselines + `build-dataset` |
| Interactive dashboard | `speechlens serve` |
| Technical documentation (max 6 pages) | `docs/TECHNICAL.md` |
| 3-10 minute video | script in `docs/VIDEO_SCRIPT.md` (to be recorded) |

## Known limitations

Synthetic evidence only; low-severity flaws are intentionally not flagged; DTW alignment may degrade
on noisy or very different speakers (use MFA via TextGrid import); English only; the rubric weights are
transparent heuristics, not calibrated to human adjudicators; the reference defines "expected", so
calibrate with several good readers. Details in `docs/TECHNICAL.md` section 9.
