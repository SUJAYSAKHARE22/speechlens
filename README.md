# SpeechLens

Contrastive speech analytics with **temporal flaw grounding**: compare a speaker's delivery
against an ideal reference of the *same text*, locate the exact regions where delivery
deviates, and explain each deviation with the acoustic evidence behind it.

> **Status:** Stage 1 of 6 complete (dataset engine). See the roadmap below.

## Quick start

```bash
pip install -e ".[dev]"
speechlens make-fixtures --out data/dev_base           # synthetic dev baseline (needs espeak-ng)
speechlens build-dataset --base data/dev_base --out data/dev_dataset
speechlens verify data/dev_dataset                     # hashes, intervals, alignment integrity
pytest                                                 # 58 tests
```

A base recording is a folder containing `audio.(wav|flac)`, `transcript.txt` and
`alignment.json`. For each one the builder emits an **ideal** reference, a **five-rung quality
ladder** (near-perfect -> botched) and **isolated single-flaw variants**, each with a JSON
label (flaw type, severity, exact start/end, word range, parameters) and its word alignment.

## How ground truth stays exact

* A recording is cut into `gap | word | gap | word ...` segments at forced-alignment
  boundaries (snapped to local energy minima). Flaws edit segments; rendering concatenates
  them and **reads labels off the final audio**, so labels are measured, not predicted.
* Untouched segments are bit-exact slices of the source (tested).
* Pitch/duration edits use Praat PSOLA (prosody changes, spectral envelope preserved).
* Inserted pauses are filled with synthesised room tone matched to the recording's own noise
  floor, so a pause is never trivially detectable as digital silence.
* Builds are bit-reproducible across processes (Praat's internal RNG is re-seeded per call).

## Flaw library (severity 0-1)

| Flaw | Acoustic signature |
|---|---|
| `rush` / `drag` | local speech rate up/down, pauses compressed/stretched |
| `misplaced_pause` | long silence where the text has no punctuation |
| `missing_pause` | silence at punctuation collapsed |
| `overlong_pause` | silence at punctuation far above expectation |
| `monotone` | F0 range compressed toward the speaker median |
| `uptalk` | terminal F0 rise on a statement |
| `trail_off` | energy decays toward the sentence end |
| `volume_collapse` | sustained energy drop |
| `mumble` | high-frequency loss, lower SNR |

## Roadmap

1. **Dataset engine** - done
2. Forced alignment + feature extraction (F0, energy, MFCC, rate, pauses; speaker-normalised)
3. Comparison, temporal grounding and causal explanation engine
4. Rubric scoring + evaluation harness (IoU vs injected ground truth, severity monotonicity)
5. Dashboard (FastAPI + Plotly/wavesurfer)
6. Documentation, dataset card, demo video
