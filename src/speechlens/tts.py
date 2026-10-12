"""Offline text-to-speech via espeak-ng with *exact* word timings.

Used for (a) development fixtures / simulated speakers and (b) the TTS-DTW aligner, which
aligns real speech to a synthetic rendering of the transcript (the approach of aeneas).
Each word is synthesised separately and joined with punctuation-dependent silences, so the
alignment is exact by construction.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from speechlens.audio import load_audio
from speechlens.schema import Alignment, WordTiming
from speechlens.text import boundary_kind, tokenize

GAP_SECONDS = {"none": 0.03, "clause": 0.35, "sentence": 0.70}


def espeak_available() -> bool:
    try:
        subprocess.run(["espeak-ng", "--version"], check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _espeak_word(word: str, pitch: int, speed: int, voice: str, sr: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "w.wav"
        subprocess.run(
            ["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch), "-w", str(wav), word],
            check=True,
            capture_output=True,
        )
        x = load_audio(wav, sr)
    keep = np.flatnonzero(np.abs(x) > 0.01 * np.max(np.abs(x)))
    return x[max(keep[0] - 40, 0) : keep[-1] + 40]


def synthesize_espeak(
    transcript: str,
    *,
    voice: str = "en-us",
    speed: int = 165,
    pitch_range: tuple[int, int] = (30, 70),
    pitch_plan: list[int] | None = None,
    gain_jitter_db: float = 3.0,
    gap_scale: float = 1.0,
    noise_floor_db: float = -52.0,
    peak: float = 0.7,
    sample_rate: int = 16_000,
    seed: int = 7,
) -> tuple[np.ndarray, Alignment]:
    """Synthesise ``transcript`` word by word; return ``(audio, exact alignment)``."""
    rng = np.random.default_rng(seed)
    noise = 10 ** (noise_floor_db / 20)

    def room(seconds: float) -> np.ndarray:
        return rng.standard_normal(int(round(seconds * sample_rate))) * noise

    tokens = tokenize(transcript)
    chunks: list[np.ndarray] = [room(0.30)]
    words: list[WordTiming] = []
    pos = len(chunks[0])
    for i, tok in enumerate(tokens):
        lo, hi = pitch_range
        pitch = pitch_plan[i] if pitch_plan is not None else int(rng.integers(lo, hi + 1))
        word = _espeak_word(tok.text, pitch, speed, voice, sample_rate)
        if gain_jitter_db:
            word = word * 10 ** (rng.uniform(-gain_jitter_db, gain_jitter_db) / 20)
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
        last = i == len(tokens) - 1
        gap = room(0.4 if last else GAP_SECONDS[boundary_kind(tok.punct_after)] * gap_scale)
        chunks += [word, gap]
        pos += len(word) + len(gap)

    audio = np.concatenate(chunks)
    audio *= peak / np.max(np.abs(audio))
    return audio, Alignment(
        words=words, sample_rate=sample_rate, duration=len(audio) / sample_rate, aligner="espeak"
    )
