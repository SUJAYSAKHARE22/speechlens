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


class Config(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config, falling back to built-in defaults."""
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as fh:
        return Config.model_validate(yaml.safe_load(fh) or {})
