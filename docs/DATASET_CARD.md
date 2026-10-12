# Dataset card: SpeechLens contrastive speech set

## Summary
Paired recordings of the **same text** delivered ideally and with injected delivery flaws on a
severity gradient, with exact flaw timestamps and word alignments. Built by
`speechlens build-dataset`; verified by `speechlens verify`.

## What ships in this repository
* `data/demo/` (12 MB): one reference reading (speaker A) and seven sample participants (speaker B)
  with ground truth, used by the dashboard.
* The full development corpus (3 simulated speakers x 16 variants = 48 recordings, 84 labelled
  flaws) is regenerated in seconds: `speechlens make-dev-corpus`. It is deterministic.
* **Real-speech data is not included** (recordings cannot be redistributed blindly; see
  `docs/SOURCING.md`). Place your baselines in `data/bases/<id>/` and build.

## Layout
```
<dataset>/index.csv                      one row per recording
<dataset>/<base_id>/transcript.txt
<dataset>/<base_id>/audio/<variant>.flac mono, 16 kHz, 16-bit
<dataset>/<base_id>/labels/<variant>.json
```
Variants: `L0_ideal`; `L1_near_perfect`, `L2_minor_slips`, `L3_mediocre`, `L4_poor`, `L5_botched`
(1, 2, 3, 5, 7 flaws); `S_<flaw>_60` (one flaw, severity 0.6, ten types).

## Label schema (`labels/*.json`)
`base_id`, `variant_id`, `kind` (ideal | ladder | single), `level` (0-5 or null), `seed`,
`sample_rate`, `duration`, `audio_sha256` (of int16 PCM), `flaw_burden`, `alignment` (word, start,
end, punctuation) and `flaws[]`: `type`, `severity` (0-1), `start`/`end` (seconds, output timeline),
`focus_start`/`focus_end` (the manipulated element, e.g. the pause), `word_start`/`word_end`, `words`,
`params` (exact parameters, e.g. `{"rate_multiplier": 1.7}`).

## Labelling method and guarantees
Labels are measured from the rendered audio, not predicted. Word times are the alignment boundaries
snapped to local energy minima (within 25 ms of the aligner's output). For pause flaws the labelled
region covers the two adjacent words plus the pause. Regions never overlap.

## Known limitations
* Ordinal level and `flaw_burden` are our construction; they are not human quality ratings.
  `flaw_burden` counts affected words, so it under-weights pause flaws; prefer the ordinal level.
* Injected flaws are idealised (clean DSP operations); human flaws are messier.
* Simulated speakers share one synthesis engine and prosody model.
* English text only.

## Intended use
Developing and stress-testing delivery-analysis systems. Not for judging real contestants
without calibration against human adjudicators.
