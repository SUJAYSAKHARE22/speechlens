"""Result schemas returned by the analysis pipeline (and the web API)."""

from __future__ import annotations

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class Evidence(BaseModel):
    metric: str
    observed: float
    expected: float
    unit: str
    delta: float = Field(description="The deviation that z was computed from, in delta_unit.")
    delta_unit: str
    sigma: float = Field(description="Natural between-speaker variation of this metric.")
    z: float = Field(description="delta / sigma (always; see rationale for the formula).")


class FlawRegion(BaseModel):
    id: int
    type: str
    family: str
    start: float
    end: float
    word_start: int
    word_end: int
    words: str
    severity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence: Evidence
    headline: str
    explanation: str
    rationale: str
    suggestion: str
    focus_start: float | None = None
    focus_end: float | None = None


class WordReport(BaseModel):
    index: int
    text: str
    start: float
    end: float
    boundary: str
    pace_log2: float
    pause_after: float
    pause_expected: float
    z: dict[str, float]


class Series(BaseModel):
    """Time-series overlay (participant time axis, 50 Hz). ``*_ref`` is the time-warped baseline."""

    t: list[float]
    f0_st: list[float | None]
    f0_st_ref: list[float | None] | None = None
    energy_db: list[float]
    energy_db_ref: list[float] | None = None


class Scores(BaseModel):
    overall: float
    categories: dict[str, float]
    preset: str
    weights: dict[str, float]


class AnalysisResult(BaseModel):
    version: str = SCHEMA_VERSION
    mode: str = Field(description="'reference' (paired baseline) or 'norm' (reference-free).")
    aligner: str
    duration: float
    reference_id: str | None
    tempo_factor: float | None
    speaker: dict[str, float]
    scores: Scores
    regions: list[FlawRegion]
    words: list[WordReport]
    series: Series
    warnings: list[str] = Field(default_factory=list)
    alignment_diagnostics: dict[str, float] = Field(default_factory=dict)
