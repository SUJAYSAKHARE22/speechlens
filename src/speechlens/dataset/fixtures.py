"""Synthetic *development* speakers built with espeak-ng.

These exist so the injection engine, evaluation and CI run offline with exact word timings.
They are NOT the released dataset: real baselines are recordings of public-domain speeches
aligned with a forced aligner. Several simulated "speakers" (different voices, speeds, pitch
registers and pause habits) read the same text, which lets us stress-test speaker-agnosticism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from speechlens.audio import save_audio
from speechlens.text import boundary_kind, tokenize
from speechlens.tts import synthesize_espeak

_FUNCTION_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "on",
        "in",
        "to",
        "for",
        "that",
        "this",
        "we",
        "our",
        "us",
        "is",
        "are",
        "be",
        "been",
        "it",
        "its",
        "as",
        "at",
        "by",
        "who",
        "so",
        "any",
        "all",
        "not",
        "can",
        "have",
        "has",
        "had",
        "here",
        "those",
        "their",
        "his",
        "her",
        "with",
        "from",
    ]
)

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


@dataclass(frozen=True)
class SimSpeaker:
    name: str
    voice: str
    speed: int
    pitch_range: tuple[int, int]
    gap_scale: float
    peak: float
    seed: int
    register: int = 0  # pitch-register offset (espeak -p units)
    jitter: float = 3.0  # speaker-specific deviation from the shared linguistic prosody


# Simulated speakers; "A" is the reference ("ideal") voice.
SPEAKERS: dict[str, SimSpeaker] = {
    s.name: s
    for s in [
        SimSpeaker("A", "en-us", 165, (30, 70), 1.00, 0.70, 7, register=0),
        SimSpeaker("B", "en-gb+f3", 150, (40, 80), 1.20, 0.45, 11, register=8),
        SimSpeaker("C", "en-gb-x-rp+m3", 185, (25, 65), 0.85, 0.85, 13, register=-4),
    ]
}


def prosody_plan(transcript: str, speaker: SimSpeaker, seed: int) -> list[int]:
    """Per-word pitch from the text (shared by all speakers) plus small speaker-specific jitter.

    Content words are stressed, function words reduced, sentence ends fall and clause ends hold:
    the linguistic structure that makes two good readings of one text resemble each other.
    """
    rng = np.random.default_rng(seed)
    plan = []
    toks = tokenize(transcript)
    for i, t in enumerate(toks):
        kind = boundary_kind(t.punct_after)
        p = 50.0 + speaker.register
        p += -6.0 if t.text.lower() in _FUNCTION_WORDS else 7.0
        p += {"sentence": -14.0, "clause": 4.0, "none": 0.0}[kind]
        p += 4.0 if i == 0 or boundary_kind(toks[i - 1].punct_after) == "sentence" else 0.0
        plan.append(int(np.clip(p + rng.normal(0.0, speaker.jitter), 5, 95)))
    return plan


def make_synthetic_base(
    out_dir: str | Path,
    transcript: str = GETTYSBURG,
    sample_rate: int = 16_000,
    seed: int = 7,
    speaker: SimSpeaker | None = None,
) -> Path:
    """Write ``audio.wav``, ``transcript.txt`` and ``alignment.json`` for a simulated speaker."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp = speaker or SPEAKERS["A"]
    audio, alignment = synthesize_espeak(
        transcript,
        voice=sp.voice,
        speed=sp.speed,
        pitch_range=sp.pitch_range,
        pitch_plan=prosody_plan(transcript, sp, seed if speaker is None else sp.seed),
        gap_scale=sp.gap_scale,
        peak=sp.peak,
        sample_rate=sample_rate,
        seed=seed if speaker is None else sp.seed,
    )
    save_audio(out / "audio.wav", audio, sample_rate)
    (out / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")
    alignment.save(out / "alignment.json")
    return out
