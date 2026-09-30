"""Transcript tokenisation.

Punctuation is *kept* and attached to the preceding word because it defines where
a fluent speaker is expected to pause (the basis of context-aware pause analysis).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

BoundaryKind = Literal["none", "clause", "sentence"]

_SENTENCE = set(".!?")
_CLAUSE = set(",;:—–-")
_LEADING = "\"'“‘([{"
_TRAILING = ".,;:!?\"'”’)]}—–-…"


@dataclass(frozen=True)
class Token:
    text: str
    punct_after: str = ""


def tokenize(transcript: str) -> list[Token]:
    """Split a transcript into words with the punctuation that follows each word."""
    spaced = re.sub(r"\s*(—|–|--)\s*", " — ", transcript.strip())
    tokens: list[Token] = []
    for raw in spaced.split():
        core = raw.lstrip(_LEADING)
        stripped = core.rstrip(_TRAILING)
        trailing = core[len(stripped) :]
        if not stripped:  # a pure-punctuation token (e.g. an em dash) belongs to the previous word
            if tokens:
                prev = tokens[-1]
                tokens[-1] = Token(prev.text, prev.punct_after + trailing)
            continue
        tokens.append(Token(stripped, trailing))
    return tokens


def boundary_kind(punct_after: str) -> BoundaryKind:
    """Classify the boundary that follows a word: sentence end, clause break or none."""
    if any(c in _SENTENCE for c in punct_after):
        return "sentence"
    if any(c in _CLAUSE for c in punct_after):
        return "clause"
    return "none"
