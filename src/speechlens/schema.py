"""Pydantic schemas for alignments and dataset labels.

These are the on-disk contract of the dataset (JSON), so they are strict and
versioned through ``DatasetConfig.dataset_version``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class WordTiming(BaseModel):
    """A transcript word with its time span in seconds."""

    index: int
    text: str
    start: float
    end: float
    punct_after: str = ""
    confidence: float | None = None

    @model_validator(mode="after")
    def _check_span(self) -> WordTiming:
        if self.end < self.start:
            raise ValueError(f"word {self.index} ({self.text!r}) ends before it starts")
        return self


class Alignment(BaseModel):
    """Word-level forced alignment of one recording."""

    words: list[WordTiming]
    sample_rate: int
    duration: float
    aligner: str = "unknown"

    @model_validator(mode="after")
    def _check_order(self) -> Alignment:
        prev_end = 0.0
        for w in self.words:
            if w.start < prev_end - 1e-6:
                raise ValueError(f"word {w.index} ({w.text!r}) overlaps its predecessor")
            prev_end = w.end
        return self

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Alignment:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class FlawLabel(BaseModel):
    """Ground truth for one injected flaw, expressed in the *output* timeline."""

    flaw_id: str
    type: str
    severity: float = Field(ge=0.0, le=1.0)
    start: float = Field(description="Start of the labelled region (s).")
    end: float = Field(description="End of the labelled region (s).")
    focus_start: float = Field(description="Start of the manipulated element (e.g. the pause).")
    focus_end: float
    word_start: int
    word_end: int
    words: str
    params: dict[str, Any] = Field(default_factory=dict)


class VariantLabel(BaseModel):
    """Full label record for one audio file in the dataset."""

    dataset_version: str
    base_id: str
    variant_id: str
    kind: str = Field(pattern="^(ideal|ladder|single)$")
    level: int | None
    level_name: str
    seed: int
    sample_rate: int
    duration: float
    audio_file: str
    audio_sha256: str = Field(description="SHA-256 of the int16 PCM samples.")
    flaw_burden: float = Field(description="sum(severity * fraction of words affected).")
    flaws: list[FlawLabel]
    alignment: Alignment

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> VariantLabel:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
