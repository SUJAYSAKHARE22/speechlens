# Demo video script (target 7-8 minutes; limit 3-10)

| time | scene | say / show |
|---|---|---|
| 0:00 | Problem | Judges score delivery subjectively; contestants get vague feedback. Show the brief in one sentence: *same text, good vs flawed delivery, find where and why*. |
| 0:30 | Idea | Diagram from the README: align both, normalise per speaker, compare word by word, explain with numbers. |
| 1:15 | Dataset: baselines | Show the baseline recordings and `alignment.json`; mention MFA import and public-domain sourcing. |
| 2:00 | Dataset: bad mirrors | Terminal: `speechlens build-dataset`; open `index.csv`; play `L0_ideal`, `L3_mediocre`, `L5_botched` back to back. Open a label JSON: *flaw type, severity, exact start/end*. Explain: labels are measured from the final audio; untouched audio is bit-exact. |
| 3:00 | Stress test | Play single flaws: rush, monotone, overlong pause. Show the severity gradient (L1 to L5). `speechlens verify` and the reproducibility claim (bit-identical rebuild). |
| 3:45 | Dashboard: good take | Start `speechlens serve`; analyse "Good reading, different speaker": different voice, no false flags, score ~90 (speaker-agnostic). |
| 4:30 | Dashboard: single flaw | "Overlong pause": click the note, audio jumps to the pause; read the explanation and open *Show the math* (delta, sigma, z). Point at the dashed ground-truth bracket vs the detected region. |
| 5:30 | Dashboard: botched | All seven flaws; scroll the notes; click through three; show the category scores and switch the speech-type preset. |
| 6:30 | Upload | Upload a real recording (yours) with its transcript against a reference; show the result. |
| 7:00 | Honesty slide | Show `docs/EVALUATION.md`: precision/recall/IoU, boundary error, and the limits (synthetic voices; run `evaluate-human` on human takes). Report the human-set numbers if you have them. |
| 7:40 | Close | Reproducibility: Docker, tests, one config; what you would do next (more speakers, MFA baselines, judge calibration). |

Tips: pre-run each sample once so the first click is instant; keep the browser at 125 % zoom;
record the terminal at a readable size; do not speed up audio playback.
