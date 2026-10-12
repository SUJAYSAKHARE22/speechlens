"""Frame- and word-level feature extraction.

Normalisation (what makes the comparison speaker-agnostic):

* **Pitch**: F0 in semitones relative to the speaker's own median; the F0 search range is
  estimated per speaker with a two-pass analysis (fewer octave errors).
* **Energy**: dB relative to the recording's own 90th-percentile level, so microphone gain and
  loudness cancel out.
* **Tempo**: handled at comparison time as a global tempo factor (see ``analysis.compare``).
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import parselmouth

from speechlens.acoustics.pitch import estimate_f0_range, f0_contour
from speechlens.alignment.refine import refine_boundaries
from speechlens.features.models import FrameFeatures, SpeakerStats, WordFeatures
from speechlens.schema import Alignment
from speechlens.text import boundary_kind, count_syllables

HOP_S = 0.01
HF_CUTOFF_HZ = 2000.0
_EPS = 1e-12


@dataclass
class SpeechFeatures:
    frames: FrameFeatures
    words: list[WordFeatures]
    stats: SpeakerStats
    alignment: Alignment


def _nearest(times: np.ndarray, grid_t: np.ndarray, step: float, n: int) -> np.ndarray:
    return np.clip(np.round((times - grid_t[0]) / step).astype(int), 0, n - 1)


def extract_frames(x: np.ndarray, sr: int) -> tuple[FrameFeatures, SpeakerStats]:
    """Compute normalised frame-level features and the speaker statistics used to normalise."""
    f0r = estimate_f0_range(x, sr)
    times, f0 = f0_contour(x, sr, f0r.floor, f0r.ceiling, HOP_S)

    # short-time energy (25 ms window) on the pitch grid, via a cumulative sum
    half = int(0.0125 * sr)
    csum = np.concatenate([[0.0], np.cumsum(x**2)])
    centre = np.round(times * sr).astype(int)
    lo, hi = np.clip(centre - half, 0, len(x)), np.clip(centre + half, 0, len(x))
    frame_pow = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)
    # relative floor (-90 dB re mean power): keeps dB values exactly gain-invariant
    db_abs = 10 * np.log10(frame_pow + 1e-9 * float(np.mean(x**2)) + 1e-30)
    ref = float(np.percentile(db_abs, 90))

    # spectral band power (FFT) for the clarity cue
    hop = int(round(HOP_S * sr))
    power = np.abs(librosa.stft(x, n_fft=512, hop_length=hop, win_length=400)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    idx = np.clip(np.round(times / HOP_S).astype(int), 0, power.shape[1] - 1)
    hf_pow, tot_pow = power[freqs >= HF_CUTOFF_HZ].sum(axis=0)[idx], power.sum(axis=0)[idx]

    # harmonics-to-noise ratio
    harm = parselmouth.Sound(np.ascontiguousarray(x), sampling_frequency=sr).to_harmonicity_cc(
        time_step=HOP_S, minimum_pitch=f0r.floor
    )
    hv = harm.values[0].astype(float)
    hv[hv < -100] = np.nan
    hnr = hv[_nearest(times, harm.xs(), HOP_S, len(hv))]

    mfcc = librosa.feature.mfcc(
        y=x, sr=sr, n_mfcc=13, n_fft=512, hop_length=hop, n_mels=40, fmax=sr / 2
    )
    mfcc = mfcc[:, np.clip(np.round(times / HOP_S).astype(int), 0, mfcc.shape[1] - 1)].T

    frames = FrameFeatures(
        times=times,
        f0_st=12 * np.log2(f0 / f0r.median),
        energy_db=db_abs - ref,
        hf_pow=hf_pow,
        tot_pow=tot_pow,
        hnr_db=hnr,
        mfcc=mfcc,
    )
    stats = SpeakerStats(
        f0_median_hz=f0r.median,
        f0_floor_hz=f0r.floor,
        f0_ceiling_hz=f0r.ceiling,
        energy_ref_db=ref,
        noise_floor_db=float(np.percentile(db_abs, 5)),
        articulation_s_per_syll=0.0,
    )
    return frames, stats


def aggregate_words(alignment: Alignment, fr: FrameFeatures) -> list[WordFeatures]:
    """Reduce frame features to one :class:`WordFeatures` per aligned word."""
    words = alignment.words
    out: list[WordFeatures] = []
    for i, w in enumerate(words):
        sl = fr.slice(w.start, w.end)
        f0 = fr.f0_st[sl]
        voiced = f0[~np.isnan(f0)]
        tail = voiced[-max(1, int(0.3 * len(voiced))) :] if len(voiced) else voiced
        hnr = fr.hnr_db[sl]
        hnr = hnr[~np.isnan(hnr)]
        tot = float(fr.tot_pow[sl].sum())
        out.append(
            WordFeatures(
                index=i,
                text=w.text,
                start=w.start,
                end=w.end,
                boundary=boundary_kind(w.punct_after),
                n_syll=count_syllables(w.text),
                pause_before=max(w.start - words[i - 1].end, 0.0) if i else w.start,
                pause_after=max(words[i + 1].start - w.end, 0.0) if i + 1 < len(words) else 0.0,
                f0_mean_st=float(voiced.mean()) if len(voiced) else float("nan"),
                f0_std_st=float(voiced.std()) if len(voiced) > 2 else float("nan"),
                f0_tail_st=float(np.median(tail)) if len(tail) else float("nan"),
                voiced_frac=float(len(voiced) / max(len(f0), 1)),
                energy_db=float(fr.energy_db[sl].mean()) if sl.stop > sl.start else float("nan"),
                hf_db=float(10 * np.log10(max(fr.hf_pow[sl].sum() / tot, 1e-10)))
                if tot > 0
                else float("nan"),
                hnr_db=float(hnr.mean()) if len(hnr) else float("nan"),
            )
        )
    return out


def extract_speech_features(
    x: np.ndarray, sr: int, alignment: Alignment, refine: bool = True
) -> SpeechFeatures:
    """Full feature set for a recording; word gaps are refined against the energy envelope."""
    frames, stats = extract_frames(x, sr)
    if refine:
        alignment = refine_boundaries(alignment, frames.times, frames.energy_db)
    words = aggregate_words(alignment, frames)
    spans = sum(w.duration for w in words)
    stats.articulation_s_per_syll = spans / max(sum(w.n_syll for w in words), 1)
    return SpeechFeatures(frames=frames, words=words, stats=stats, alignment=alignment)
