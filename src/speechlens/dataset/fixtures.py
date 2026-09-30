"""Synthetic *development* baselines built with espeak-ng.

These exist so the injection engine, tests and CI run offline with exact word
timings. They are NOT part of the released dataset: real baselines are recordings
of public-domain speeches aligned with a forced aligner (see the alignment stage).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from speechlens.audio import load_audio, save_audio
from speechlens.schema import Alignment, WordTiming
from speechlens.text import boundary_kind, tokenize

# Abraham Lincoln, Gettysburg Address (1863) - public domain.
GETTYSBURG = (
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, "
    "conceived in Liberty, and dedicated to the proposition that all men are created equal. "
    "Now we are engaged in a great civil war, testing whether that nation, or any nation so "
    "conceived and so dedicated, can long endure. We are met on a great battle-field of that war. "
    "We have come to dedicate a portion of that field, as a final resting place for those who "
    "here gave their lives that that nation might live. It is altogether fitting and proper "
    "that we should do this. But, in a larger sense, we can not dedicate, we can not consecrate, "
    "we can not hallow, this ground. The brave men, living and dead, who struggled here, have "
    "consecrated it, far above our poor power to add or detract."
)

_GAP_SECONDS = {"none": 0.03, "clause": 0.35, "sentence": 0.70}


def _espeak_word(word: str, pitch: int, speed: int, sr: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "w.wav"
        subprocess.run(
            ["espeak-ng", "-v", "en-us", "-s", str(speed), "-p", str(pitch), "-w", str(wav), word],
            check=True,
            capture_output=True,
        )
        x = load_audio(wav, sr)
    keep = np.flatnonzero(np.abs(x) > 0.01 * np.max(np.abs(x)))
    return x[max(keep[0] - 40, 0) : keep[-1] + 40]


def make_synthetic_base(
    out_dir: str | Path,
    transcript: str = GETTYSBURG,
    sample_rate: int = 16_000,
    seed: int = 7,
    noise_floor_db: float = -52.0,
) -> Path:
    """Synthesise a base recording (word by word) plus exact alignment and transcript."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    noise = 10 ** (noise_floor_db / 20)

    def room(seconds: float) -> np.ndarray:
        return rng.standard_normal(int(round(seconds * sample_rate))) * noise

    tokens = tokenize(transcript)
    chunks: list[np.ndarray] = [room(0.30)]
    words: list[WordTiming] = []
    pos = len(chunks[0])
    for i, tok in enumerate(tokens):
        # Varied per-word pitch gives the contour real dynamics to be flattened later.
        word = _espeak_word(tok.text, int(rng.integers(30, 71)), 165, sample_rate)
        word *= 10 ** (rng.uniform(-3, 3) / 20)
        edge = min(80, len(word) // 4)
        word[:edge] *= np.linspace(0, 1, edge)
        word[-edge:] *= np.linspace(1, 0, edge)
        words.append(
            WordTiming(
                index=i,
                text=tok.text,
                start=pos / sample_rate,
                end=(pos + len(word)) / sample_rate,
                punct_after=tok.punct_after,
            )
        )
        gap = room(_GAP_SECONDS[boundary_kind(tok.punct_after)] if i < len(tokens) - 1 else 0.4)
        chunks += [word, gap]
        pos += len(word) + len(gap)

    audio = np.concatenate(chunks)
    peak = np.max(np.abs(audio))
    audio *= 0.7 / peak
    save_audio(out / "audio.wav", audio, sample_rate)
    (out / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")
    Alignment(
        words=words, sample_rate=sample_rate, duration=len(audio) / sample_rate, aligner="synthetic"
    ).save(out / "alignment.json")
    return out
