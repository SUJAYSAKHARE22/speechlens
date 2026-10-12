"""Typed, validated configuration.

Everything that influences dataset content lives here so that a run is fully
described by (config file, seed, input recordings) and is therefore reproducible.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    sample_rate: int = 16_000
    snap_window_ms: float = Field(
        25.0, description="Cut points snap to the energy minimum within +-this window."
    )
    edge_fade_ms: float = Field(
        3.0, description="Fade applied to the edges of modified word segments."
    )
    context_ms: float = Field(
        150.0, description="Context kept around a region during PSOLA editing."
    )


class LevelSpec(BaseModel):
    """One rung of the quality ladder (0 = ideal ... 5 = botched)."""

    name: str
    n_flaws: int
    severity: tuple[float, float]
    types: list[str] | None = Field(None, description="Restrict the flaw pool for this level.")


def _default_levels() -> dict[int, LevelSpec]:
    subtle = ["rush", "drag", "trail_off", "missing_pause", "uptalk"]
    return {
        0: LevelSpec(name="ideal", n_flaws=0, severity=(0.0, 0.0)),
        1: LevelSpec(name="near_perfect", n_flaws=1, severity=(0.20, 0.30), types=subtle),
        2: LevelSpec(name="minor_slips", n_flaws=2, severity=(0.30, 0.50)),
        3: LevelSpec(name="mediocre", n_flaws=3, severity=(0.50, 0.70)),
        4: LevelSpec(name="poor", n_flaws=5, severity=(0.70, 0.90)),
        5: LevelSpec(name="botched", n_flaws=7, severity=(0.90, 1.00)),
    }


class DatasetConfig(BaseModel):
    dataset_version: str = "0.1.0"
    seed: int = 1337
    levels: dict[int, LevelSpec] = Field(default_factory=_default_levels)
    single_flaw_severities: list[float] = [0.6]
    audio_format: str = Field("flac", pattern="^(flac|wav)$")


class Tolerance(BaseModel):
    """Natural between-speaker variation per metric (the scale of a z-score).

    Deviations are only flagged when they exceed several multiples of this normal variation.
    Defaults are deliberately conservative; ``speechlens calibrate`` can only widen them.
    """

    pace_sigma: float = Field(0.20, description="log2 duration ratio (~15 %).")
    pause_sigma_abs: float = Field(0.12, description="seconds.")
    pause_sigma_rel: float = Field(0.25, description="fraction of the expected pause.")
    pitch_sigma: float = Field(0.25, description="log2 ratio of windowed F0 std.")
    terminal_sigma: float = Field(1.5, description="semitones.")
    energy_sigma: float = Field(3.0, description="dB.")
    clarity_sigma: float = Field(3.0, description="dB of high-frequency share.")


class DetectionConfig(BaseModel):
    z_threshold: float = 2.5
    pause_min_excess_s: float = 0.25
    missing_pause_ratio: float = 0.5
    missing_pause_min_expected_s: float = 0.15
    missing_pause_min_loss_s: float = 0.15
    min_words_pace: int = 3
    min_words_pitch: int = 5
    min_words_energy: int = 3
    min_words_clarity: int = 2
    max_gap_words: int = 1
    trail_window: int = 5
    pitch_half_window: int = 2
    z_cap: float = 6.0
    score_gain: float = 12.0


class AnalysisConfig(BaseModel):
    tolerance: Tolerance = Field(default_factory=Tolerance)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)


class Config(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config, falling back to built-in defaults."""
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as fh:
        return Config.model_validate(yaml.safe_load(fh) or {})
