# Sourcing real recordings and building a non-circular test set

The simulated speakers prove the machinery works. **Winning evidence needs human speech.**

## 1. Baselines ("ideal" readings)
Choose short passages (60-120 s, at least 120 words, so the top ladder level fits) with clear
punctuation. Prefer material you can legally redistribute:

* **Safest:** texts in the public domain that *you read yourself* or that a teammate records (e.g.
  the Gettysburg Address, FDR's 1933 inaugural, Lincoln's Second Inaugural). The text and your
  recording are both yours to publish. Record several *good* readers per text: they become the
  calibration set and a more robust reference.
* **Existing recordings:** the text of a speech and a particular recording are copyrighted
  separately. Government works (e.g. US federal speeches) are often free to reuse; many famous
  speeches (a well-known 1963 civil-rights speech, for instance) are **still under copyright**.
  TED talks are CC BY-NC-ND (no derivatives), which conflicts with distributing modified copies.
  Check each item; cite the licence in the dataset card; if in doubt, record your own reading.
* Record in a quiet room, one speaker, 16 kHz or higher, no music, no clipping.

## 2. Aligning a baseline
Best quality: run the Montreal Forced Aligner externally and import the TextGrid
(`speechlens.alignment.alignment_from_textgrid(...).save("alignment.json")`). No MFA? Use the TTS-DTW
aligner (`TTSDTWAligner`) as a quick approximation and spot-check boundaries by ear.
Folder layout expected by the builder:
```
data/bases/<id>/audio.wav        transcript.txt (with punctuation)        alignment.json
```
Then: `speechlens build-dataset --base data/bases --out data/dataset && speechlens verify data/dataset`.

## 3. The human test set (this is what removes the "circular validation" objection)
1. Have 3-4 people read each baseline text: some flawlessly, some with *deliberate, natural* flaws
   (rushing a phrase, going flat, pausing mid-sentence, trailing off, mumbling, rising endings).
2. Annotate each flaw's span in an audio editor (Audacity labels work) using our flaw names:
   `rush drag misplaced_pause missing_pause overlong_pause monotone trail_off volume_collapse mumble uptalk`.
3. Save as `annotations.json` next to the audio:
```json
{"recordings": [
  {"audio": "alex_rushed.wav", "flaws": [{"type": "rush", "start": 12.1, "end": 14.0}]},
  {"audio": "sam_good.wav", "flaws": []}
]}
```
4. Run `speechlens evaluate-human --reference data/bases/<id> --annotations annotations.json`.
   Report these numbers *separately* from the synthetic ones, including the gap between them.
5. Calibrate on the flawless takes: `speechlens calibrate --reference data/bases/<id> --good good1.wav good2.wav`
   then pass the resulting YAML with `--config`.

Annotate before looking at the system's output, and keep calibration takes out of the test set.
