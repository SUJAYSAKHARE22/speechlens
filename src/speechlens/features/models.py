"""Plain data containers for acoustic features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeakerStats:
    """Per-recording statistics used to normalise features (speaker/channel agnostic)."""

    f0_median_hz: float
    f0_floor_hz: float
    f0_ceiling_hz: float
    energy_ref_db: float  # 90th-percentile frame level; 0 dB after normalisation
    noise_floor_db: float  # 5th-percentile frame level (absolute)
    articulation_s_per_syll: float  # mean word-span seconds per syllable


@dataclass
class FrameFeatures:
    """Frame-level contours on a common 10 ms grid (``times`` are frame centres, seconds)."""

    times: np.ndarray
    f0_st: np.ndarray  # semitones re speaker median; NaN where unvoiced
    energy_db: np.ndarray  # dB re the recording's 90th-percentile level
    hf_pow: np.ndarray  # linear power above 2 kHz
    tot_pow: np.ndarray  # linear total power (FFT frame)
    hnr_db: np.ndarray  # Praat harmonicity (NaN where unvoiced)
    mfcc: np.ndarray  # (frames, 13)

    def slice(self, t0: float, t1: float) -> slice:
        return slice(int(np.searchsorted(self.times, t0)), int(np.searchsorted(self.times, t1)))


@dataclass
class WordFeatures:
    index: int
    text: str
    start: float
    end: float
    boundary: str  # none | clause | sentence  (punctuation after the word)
    n_syll: int
    pause_before: float
    pause_after: float
    f0_mean_st: float
    f0_std_st: float
    f0_tail_st: float  # median of the last 30 % of voiced frames (terminal pitch)
    voiced_frac: float
    energy_db: float  # mean frame level inside the word, dB re recording ref
    hf_db: float  # 10*log10(power above 2 kHz / total power), word-aggregated
    hnr_db: float

    @property
    def duration(self) -> float:
        return self.end - self.start
