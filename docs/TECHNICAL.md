# SpeechLens: technical documentation

*Track C: contrastive speech analytics and temporal flaw grounding.*

## 1. Approach in one paragraph

A good recording of a text is the **reference**. A participant reads the same text. Both are
force-aligned to the transcript, so every word has a time span in each recording. Acoustic
features are extracted and **normalised per speaker** (pitch in semitones relative to the speaker's
own median, loudness relative to the recording's own level, tempo as a global factor). Because
the words correspond one-to-one, the comparison is done **word by word**, not frame by frame:
for each word we compute how far the participant departs from the expectation, in units of the
natural between-speaker variation (a z-score). Runs of words that exceed a threshold become
**flaw regions** with word-boundary-precise start/end times. Each region is explained with a
template that quotes the underlying numbers. A five-category rubric turns the deviations into scores.

```
 audio + transcript ──► alignment ──► features ──► deviation (per word, z-scores) ──► detectors
        │                 (DTW / MFA)  (F0, E, HF…)      reference | norm mode           │
        └─ reference ─────────┘                                                          ▼
                                                      explanations ◄── flaw regions ──► rubric scores
                                                              └────────► dashboard (overlay + notes)
```

## 2. Dataset construction

**Baselines ("ideal").** Real recordings of public-domain texts, force-aligned (section 3). The
repository also ships three *simulated* speakers (espeak-ng voices with different registers,
tempos and pause habits that share a text-driven prosody model) so everything runs offline. They
exist to test the machinery and **are not a substitute for human speech**.

**Bad mirrors by injection.** Rather than record every flaw by hand, the engine takes a baseline
and a word alignment and injects ten parametric flaws at a severity *s* in (0, 1]. This gives a
controllable gradient from imperceptible to egregious, and **exact ground-truth labels for free**.

| flaw | operation (severity *s*) | acoustic signature |
|---|---|---|
| rush / drag | PSOLA duration x 1/(1+0.9s) or x (1+s); inter-word gaps shrunk/stretched | local rate up / down |
| misplaced pause | +0.25+1.35s s of room tone where the text has no punctuation | silence mid-phrase |
| overlong pause | +0.4+2.0s s at punctuation | silence >> expectation |
| missing pause | gap at punctuation scaled by (1-s) | silence collapsed |
| monotone | F0 contour compressed toward the speaker median by factor (1-s) in semitones | F0 range down |
| uptalk | terminal pitch ramp up to +7s semitones | final F0 rise |
| trail-off | gain ramp to -(6+18s) dB across the last 3-5 words | energy decay |
| volume collapse | constant -(8+16s) dB | sustained energy drop |
| mumble | low-pass at 6500-5300s Hz plus band-limited noise | HF loss, lower SNR |

**Why labels are exact.** A recording is cut into `gap | word | gap | word …` segments at the
alignment boundaries (snapped to local energy minima). Injectors replace the audio of segments;
rendering concatenates them and **reads each label off the final audio**. Untouched segments are
bit-exact slices of the source (tested), so nothing outside a labelled region changes.

**Realism safeguards.** Pitch and duration edits use Praat PSOLA (prosody changes, spectral
envelope preserved), applied with surrounding context to avoid edge artefacts. Inserted pauses are
filled with *synthesised room tone matched to the recording's own noise floor*, so a pause is not
trivially detectable as digital silence. Uptalk is only placed on words that are voiced
enough to carry a pitch rise. Praat's internal random generator is re-seeded before every resynthesis:
without this, stretched audio differed from run to run (found and fixed during development), and
the build is now **bit-identical across processes**.

**Ladder and variants.** For every baseline: `L0` ideal; `L1`..`L5` composites with 1, 2, 3, 5, 7
flaws at rising severity ("near perfect" to "botched"); and one isolated variant per flaw type
(severity 0.6). Regions never overlap and are separated by at least one word. Each file has a
JSON label (type, severity, start, end, word range, parameters, word alignment, audio SHA-256);
`speechlens verify` re-checks hashes, intervals and alignment integrity.

## 3. Forced alignment

* **Reference-DTW** (default): align the recording to the reference recording (MFCC c1-c12 plus
  log-energy, per-utterance mean/variance normalised) and project the reference's word boundaries
  through the warping path. Needs no model downloads; deterministic.
* **TTS-DTW**: for text without a reference recording, align to an espeak-ng rendering of the transcript.
* **MFA import**: gold-standard baselines come from the Montreal Forced Aligner; its TextGrid is imported.
* **Boundary refinement.** DTW boundaries are re-derived from the energy envelope: a gap is the
  longest low-energy run near the aligned boundary. The silence threshold is *local* (30 dB below
  the local peak, never below noise floor + 3 dB), otherwise quiet passages are mistaken for silence
  and their words get clipped (a bug found by inspecting a false positive, now covered by a test).

Measured on the final dev corpus (cross-speaker, including botched takes): median boundary error
7-10 ms, P99 37-60 ms, 99% within 50 ms. A high DTW cost per step raises a warning that the audio
may not match the transcript.

## 4. Features and speaker-agnostic normalisation

Frames use a 10 ms grid: F0 (Praat, two-pass range estimation per speaker), short-time energy,
harmonics-to-noise ratio, power above 2 kHz vs total (FFT), and 13 MFCCs. Word-level aggregates:
duration, syllables (heuristic), pause before/after, F0 mean/std/terminal value, voiced fraction,
energy, high-frequency share, HNR.

| quantity | normalisation | what it removes |
|---|---|---|
| F0 | semitones re the speaker's median | register, gender |
| energy | dB re the recording's 90th percentile | gain, microphone, distance |
| high-frequency share | centred on the recording's median | timbre, channel |
| tempo | global factor *s* = median(duration_you / duration_ref) | naturally fast or slow speakers |

Energy features are exactly gain-invariant (tested). Terminal-pitch claims are only made on words
with enough voiced signal; a word ending in unvoiced consonants gives a meaningless "tail".

## 5. Comparison, detection and explanation

Per word *i*, deviations are expressed so that `z = delta / sigma`, where sigma is the natural
between-speaker variation of that metric:

| metric | delta | sigma (default) |
|---|---|---|
| pace | log2[(d_you / d_ref) / s] | 0.20 octaves |
| pause after word | observed - s x reference pause | 0.12 s + 0.25 x expected |
| pitch range | log2(F0 std_you / std_ref), 5-word window | 0.25 octaves |
| terminal pitch | tail F0_you - tail F0_ref (sentence ends) | 1.5 st |
| energy | E_you - E_ref | 3 dB |
| clarity | HF share_you - HF share_ref (median-centred) | 3 dB |

**Detection.** A region is a run of consecutive words (one-word gaps allowed) with |z| >= 2.5 on
its metric; pacing uses a 3-word median. Pauses are judged *against the punctuation the text
provides*: a long silence is "misplaced" if there is no punctuation and "overlong" if there is;
"missing" if it falls to <= 50 % of the expected length. A fade at a sentence end is *trail-off*;
a steady drop elsewhere is *volume collapse*. Pause regions already explained by a pacing region
that contains them (rushing squeezes the pauses inside it) are not reported twice.

**Tolerance calibration.** `speechlens calibrate` measures the natural spread of good readings by
other speakers and sets sigma so that their 95th-percentile deviation maps to |z| = 1.5. It can
only widen the defaults, so a tiny calibration set cannot make the detector over-sensitive.

**Reference-free (norm) mode.** Without a reference, expectations come from the speaker's own
median level and articulation rate, punctuation-class pause norms and population pitch variability.
It uses the same detectors and is labelled "lower confidence".

**Causal explanation.** Each region carries: the flaw type and family; word-boundary start/end
(plus the pause interval for pause flaws); severity and confidence; an *evidence* record
(observed, expected, delta, sigma, z); a plain-language explanation; the **mathematical rationale**
(formula with the actual numbers); and a concrete suggestion. Templates are deterministic. A test
enforces that `z == delta / sigma` and that the rationale quotes exactly that z.

## 6. Rubric and scoring

Five categories: **pacing, pauses, pitch, energy, clarity**, each `100 x exp(-12 D)` where D is the
mean over words of min(max(|z| - 1, 0), 6) / 6: free within one natural-variation unit, smooth
beyond it, and sensitive to both *how much* of the speech deviates and *how far*. Pauses and rising
final words are *point events* measured on one word, so their penalty is spread over about five
words (otherwise a 2-second misplaced pause barely registered). Only a loss of clarity or a flat or
rising pitch counts as a flaw. The overall score is a weighted mean; presets (`balanced`,
`interpretive`, `declamation`, `extemporaneous`, `persuasive`) shift weight toward pitch, energy or
pacing/pauses. **Weights and scale are transparent heuristics, not calibrated to human judges.**

## 7. Evaluation (see `docs/EVALUATION.md`)

Design: reference voice A against two other simulated speakers (16 recordings each), matching
detected to labelled regions by time-IoU. Results (main run): precision 1.0 / 0.88, recall
0.82 / 0.75, matched IoU 0.91 / 0.84, type accuracy 100 %, median start/end error 12-18 ms; flawless
takes by other speakers yield zero false regions; score falls monotonically down the ladder
(Spearman -1.0) and the correct rubric category loses the most points in 90 % of single-flaw
cases. Recall is 1.0 for high-severity and 0.0 for low-severity flaws (by design).
**Caveat:** synthetic speakers, flaws injected by our own engine, and several constants tuned on
the same recordings. `speechlens evaluate-human` runs the same metrics on human takes with
hand-annotated flaw spans: that is the test that matters and the repository is ready for it.

## 8. Architecture and reproducibility

`dataset/` (injection engine, builder, verifier) · `alignment/` · `features/` · `analysis/`
(compare, detect, explain, rubric, calibration, pipeline) · `evaluation/` · `api/` (FastAPI +
single-file dashboard with a vendored Plotly, no CDN) · `cli.py`. All thresholds live in one
validated config (`configs/default.yaml`). Seeds are derived per variant from a hash of
(seed, base, variant). The suite has 118 tests (injection correctness, determinism, alignment,
detection, explanation consistency, API); a Dockerfile and CI workflow are provided.

## 9. Limitations

Synthetic evidence only; DTW alignment may degrade on very noisy or very different speakers
(use MFA); syllable counts are heuristic and English-only; the rubric is uncalibrated; pitch
tracking can fail on creaky voice; low-severity flaws are not flagged; the reference defines
"expected", so a reference with idiosyncratic delivery biases results (calibrate with several
good readers); no semantic or content-quality assessment (this is delivery analysis).
