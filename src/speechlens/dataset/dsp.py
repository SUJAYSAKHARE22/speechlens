"""Signal-processing primitives used to inject delivery flaws.

Pitch and duration edits use Praat's PSOLA resynthesis (via Parselmouth): it keeps
the spectral envelope intact, so an edit changes *prosody only* rather than adding
obvious vocoder artefacts that a detector could latch onto instead of the flaw.

Every region edit is done with surrounding context and then cut back out, which
avoids PSOLA edge effects and keeps splice points clean.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import butter, sosfiltfilt, welch

from speechlens.acoustics.pitch import F0Range
from speechlens.audio import EPS, rms

# (absolute times in the source recording [s], f0 [Hz]) -> edited f0 [Hz]
PitchEdit = Callable[[np.ndarray, np.ndarray], np.ndarray]

# Praat draws from an internal random generator when it time-stretches (pulse placement in
# unvoiced stretches). Left unseeded, duration edits differ from run to run, which would break
# dataset reproducibility. We therefore re-seed Praat before *every* resynthesis, so the output
# depends only on the input audio and parameters -- not on call order or process history.
_PRAAT_SEED = 20_260_930


def _seed_praat(seed: int = _PRAAT_SEED) -> None:
    parselmouth.praat.run(f"random_initializeWithSeedUnsafelyButPredictably ({seed})")


def _resynthesize(
    x: np.ndarray,
    sr: int,
    f0: F0Range,
    duration_factor: float,
    pitch_edit: PitchEdit | None,
    t_offset: float,
) -> np.ndarray:
    _seed_praat()
    snd = parselmouth.Sound(np.ascontiguousarray(x), sampling_frequency=sr)
    manip = call(snd, "To Manipulation", 0.01, f0.floor, f0.ceiling)

    if pitch_edit is not None:
        tier = call(manip, "Extract pitch tier")
        n = int(call(tier, "Get number of points"))
        if n > 0:
            t = np.array([call(tier, "Get time from index", i) for i in range(1, n + 1)])
            v = np.array([call(tier, "Get value at index", i) for i in range(1, n + 1)])
            v_new = np.clip(pitch_edit(t + t_offset, v), 40.0, 900.0)
            new_tier = call("Create PitchTier", "edited", snd.xmin, snd.xmax)
            for ti, vi in zip(t, v_new, strict=True):
                call(new_tier, "Add point", float(ti), float(vi))
            call([new_tier, manip], "Replace pitch tier")

    if abs(duration_factor - 1.0) > 1e-9:
        dur = call("Create DurationTier", "dur", snd.xmin, snd.xmax)
        call(dur, "Add point", snd.xmin, float(duration_factor))
        call(dur, "Add point", snd.xmax, float(duration_factor))
        call([dur, manip], "Replace duration tier")

    return np.asarray(call(manip, "Get resynthesis (overlap-add)").values[0], dtype=float)


def edit_region(
    source: np.ndarray,
    sr: int,
    a: int,
    b: int,
    *,
    f0: F0Range,
    duration_factor: float = 1.0,
    pitch_edit: PitchEdit | None = None,
    context: int = 2400,
) -> np.ndarray:
    """Return source[a:b] after PSOLA duration/pitch editing.

    ``duration_factor`` > 1 lengthens (slows), < 1 shortens (speeds up). The output
    length is exactly ``round((b - a) * duration_factor)`` samples.
    """
    if b <= a:
        return source[a:b].copy()
    if abs(duration_factor - 1.0) < 1e-9 and pitch_edit is None:
        return source[a:b].copy()

    c0, c1 = max(0, a - context), min(len(source), b + context)
    y = _resynthesize(source[c0:c1], sr, f0, duration_factor, pitch_edit, c0 / sr)
    lo = int(round((a - c0) * duration_factor))
    hi = int(round((b - c0) * duration_factor))
    out = y[lo:hi]
    target = int(round((b - a) * duration_factor))
    if len(out) < target:
        out = np.pad(out, (0, target - len(out)))
    return out[:target].copy()


def lowpass_region(
    source: np.ndarray, sr: int, a: int, b: int, cutoff_hz: float, context: int = 2400
) -> np.ndarray:
    """Zero-phase Butterworth low-pass of source[a:b] (with context to avoid edge ringing)."""
    c0, c1 = max(0, a - context), min(len(source), b + context)
    sos = butter(4, min(cutoff_hz, sr / 2 - 100) / (sr / 2), btype="low", output="sos")
    return sosfiltfilt(sos, source[c0:c1])[a - c0 : b - c0].copy()


def add_noise_at_snr(
    x: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    sr: int | None = None,
    cutoff_hz: float | None = None,
) -> np.ndarray:
    """Add noise so the result has the requested SNR; optionally band-limit it to ``cutoff_hz``.

    Band-limiting matters for muffling effects: broadband noise would *add* high-frequency
    energy and undo the very low-pass we are trying to simulate.
    """
    noise = rng.standard_normal(len(x))
    if cutoff_hz is not None and sr is not None:
        sos = butter(4, min(cutoff_hz, sr / 2 - 100) / (sr / 2), btype="low", output="sos")
        noise = sosfiltfilt(sos, noise)
    noise *= rms(x) / (10.0 ** (snr_db / 20.0)) / (rms(noise) + EPS)
    return x + noise


class RoomTone:
    """Synthesises noise that matches a recording's own noise floor.

    Padding an inserted pause with digital silence would make it trivially
    detectable (and unrealistic). We instead estimate the long-term spectrum of the
    quietest frames and synthesise fresh noise with that spectrum and level.
    """

    def __init__(self, freqs: np.ndarray, amplitude: np.ndarray, level_rms: float, sr: int):
        self._freqs, self._amp, self._level, self._sr = freqs, amplitude, level_rms, sr

    @classmethod
    def from_audio(cls, x: np.ndarray, sr: int, quantile: float = 0.10) -> RoomTone:
        frame = int(0.02 * sr)
        n = len(x) // frame
        frames = x[: n * frame].reshape(n, frame)
        energy = np.sqrt(np.mean(frames**2, axis=1))
        quiet = frames[energy <= np.quantile(energy, quantile)].ravel()
        if len(quiet) < 256 or rms(quiet) < 1e-7:
            return cls(np.array([0.0, sr / 2]), np.array([1.0, 1.0]), 0.0, sr)
        freqs, psd = welch(quiet, fs=sr, nperseg=512)
        return cls(freqs, np.sqrt(psd), rms(quiet), sr)

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0 or self._level == 0.0:
            return np.zeros(max(n, 0))
        spec = np.fft.rfft(rng.standard_normal(n))
        shape = np.interp(np.fft.rfftfreq(n, 1.0 / self._sr), self._freqs, self._amp)
        y = np.fft.irfft(spec * shape, n)
        return y * (self._level / (rms(y) + EPS))


def shrink_gap(gap: np.ndarray, new_len: int, fade: int = 48) -> np.ndarray:
    """Shorten a pause by removing samples from its middle (keeps both decay and onset)."""
    if new_len >= len(gap):
        return gap.copy()
    new_len = max(new_len, 0)
    head = int(round(new_len * 0.6))
    tail = new_len - head
    out = np.concatenate([gap[:head], gap[len(gap) - tail :] if tail else gap[:0]])
    f = min(fade, head, tail)
    if f > 1:  # smooth the seam between the two kept parts
        ramp = np.linspace(0.0, 1.0, f)
        out[head - f : head] *= ramp[::-1]
        out[head : head + f] *= ramp
    return out


def extend_gap(
    gap: np.ndarray, extra: int, tone: RoomTone, rng: np.random.Generator, fade: int = 96
) -> np.ndarray:
    """Lengthen a pause by inserting synthesised room tone between its head and tail."""
    if extra <= 0:
        return gap.copy()
    head_n = min(len(gap), int(0.05 * tone._sr))
    tail_n = min(len(gap) - head_n, int(0.03 * tone._sr))
    head, tail = gap[:head_n].copy(), gap[len(gap) - tail_n :].copy()
    middle = tone.generate(len(gap) - head_n - tail_n + extra, rng)
    f = min(fade, len(middle) // 2)
    if f > 1:
        ramp = np.linspace(0.0, 1.0, f)
        if len(head):
            head[-min(f, len(head)) :] *= ramp[::-1][: min(f, len(head))]
        middle[:f] *= ramp
        middle[-f:] *= ramp[::-1]
        if len(tail):
            tail[: min(f, len(tail))] *= ramp[: min(f, len(tail))]
    return np.concatenate([head, middle, tail])
