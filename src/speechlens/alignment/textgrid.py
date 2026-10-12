"""Import word alignments from Praat TextGrids (e.g. Montreal Forced Aligner output).

MFA is the most accurate open forced aligner; run it externally, then import its TextGrid
here so the rest of the pipeline can use gold-quality baseline alignments.
"""

from __future__ import annotations

import re
from pathlib import Path

from speechlens.schema import Alignment, WordTiming
from speechlens.text import tokenize

_INTERVAL = re.compile(
    r"xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*\"((?:[^\"]|\"\")*)\""
)
_SILENCE = {"", "sp", "sil", "spn", "<eps>", "<unk>_sil"}


def _words_tier(text: str, tier_name: str) -> str:
    chunks = re.split(r"\n\s*item \[\d+\]:", text)
    for chunk in chunks:
        m = re.search(r"name\s*=\s*\"([^\"]*)\"", chunk)
        if m and m.group(1).lower() == tier_name.lower():
            return chunk
    raise ValueError(f"no tier named {tier_name!r} in TextGrid")


def alignment_from_textgrid(
    path: str | Path, transcript: str, sample_rate: int, tier: str = "words"
) -> Alignment:
    """Read a long-format TextGrid and map its word intervals onto the transcript tokens."""
    raw = Path(path).read_text(encoding="utf-8")
    spans = [
        (float(a), float(b), t.replace('""', '"'))
        for a, b, t in _INTERVAL.findall(_words_tier(raw, tier))
    ]
    spoken = [(a, b, t) for a, b, t in spans if t.strip().lower() not in _SILENCE]
    tokens = tokenize(transcript)
    if len(spoken) != len(tokens):
        raise ValueError(
            f"TextGrid has {len(spoken)} word intervals but the transcript has {len(tokens)} words"
        )
    words = [
        WordTiming(index=i, text=tok.text, start=a, end=b, punct_after=tok.punct_after)
        for i, ((a, b, _), tok) in enumerate(zip(spoken, tokens, strict=True))
    ]
    duration = max((b for _, b, _ in spans), default=0.0)
    return Alignment(
        words=words, sample_rate=sample_rate, duration=duration, aligner="mfa-textgrid"
    )
