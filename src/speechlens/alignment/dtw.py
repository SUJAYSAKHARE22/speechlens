"""DTW-based forced alignment.

Two backends share one engine:

* :class:`ReferenceDTWAligner` - align a recording to a *reference recording of the same text*
  whose word timings are known, then project the reference word boundaries through the
  warping path. This is the primary path (a paired good baseline always exists).
* :class:`TTSDTWAligner` - for text without a reference recording, synthesise the transcript
  with espeak-ng (exact timings) and use that as the reference.

Neither needs model downloads, and both are deterministic. They are less precise than an
HMM/CTC aligner such as the Montreal Forced Aligner (see ``alignment.textgrid`` to import MFA
output); boundaries are therefore refined against the energy envelope afterwards.
"""

from __future__ import annotations

import librosa
import numpy as np

from speechlens.schema import Alignment, WordTiming
from speechlens.tts import synthesize_espeak

HOP_S = 0.02
# additive penalty for non-diagonal steps; keeps the path close to the diagonal but still lets it
# absorb long inserted pauses / large tempo changes
_STEP_PENALTY = (0.0, 1.0, 1.0)


def alignment_features(x: np.ndarray, sr: int, hop_s: float = HOP_S) -> np.ndarray:
    """MFCC(c1-c12) + log-energy, per-utterance mean/variance normalised -> (d, frames).

    CMVN removes channel and gross speaker-timbre differences so the path follows phonetic
    content; the energy row anchors silence against silence.
    """
    hop = int(round(hop_s * sr))
    mfcc = librosa.feature.mfcc(
        y=x, sr=sr, n_mfcc=13, n_fft=512, hop_length=hop, n_mels=40, fmax=sr / 2
    )[1:]
    rms = librosa.feature.rms(y=x, frame_length=512, hop_length=hop)[0]
    n = min(mfcc.shape[1], len(rms))
    feats = np.vstack([mfcc[:, :n], 20 * np.log10(rms[:n] + 1e-6)[None, :]])
    feats = (feats - feats.mean(axis=1, keepdims=True)) / (feats.std(axis=1, keepdims=True) + 1e-8)
    feats[-1] *= 1.5  # emphasise the speech/silence cue
    return feats.astype(np.float32)


class ReferenceDTWAligner:
    """Align recordings to a reference recording with known word boundaries."""

    name = "dtw-reference"

    def __init__(self, ref_audio: np.ndarray, ref_alignment: Alignment, sr: int):
        self.sr = sr
        self.ref = ref_alignment
        self._ref_feats = alignment_features(ref_audio, sr)

    def align(self, audio: np.ndarray) -> Alignment:
        feats = alignment_features(audio, self.sr)
        cost, wp = librosa.sequence.dtw(
            X=self._ref_feats,
            Y=feats,
            metric="euclidean",
            weights_add=np.array(_STEP_PENALTY),
            subseq=False,
        )
        path = wp[::-1]  # (ref_frame, participant_frame), start -> end
        n_ref, n_par = self._ref_feats.shape[1], feats.shape[1]

        def project(t_ref: float) -> float:
            j = int(np.clip(round(t_ref / HOP_S), 0, n_ref - 1))
            idx = np.flatnonzero(path[:, 0] == j)
            i = (
                float(np.median(path[idx, 1]))
                if len(idx)
                else float(np.interp(j, path[:, 0], path[:, 1]))
            )
            return min(i, n_par - 1) * HOP_S

        duration = len(audio) / self.sr
        words: list[WordTiming] = []
        prev_end = 0.0
        for w in self.ref.words:
            s = max(project(w.start), prev_end)
            e = min(max(project(w.end), s), duration)
            words.append(w.model_copy(update={"start": s, "end": e, "confidence": None}))
            prev_end = e
        diag = {
            "dtw_cost_per_step": float(cost[-1, -1] / len(path)),
            "duration_ratio": float(duration / max(self.ref.duration, 1e-9)),
        }
        return Alignment(
            words=words, sample_rate=self.sr, duration=duration, aligner=self.name, diagnostics=diag
        )


class TTSDTWAligner:
    """Align a recording to an espeak-ng rendering of the transcript (no reference audio)."""

    name = "dtw-tts"

    def __init__(self, sr: int, voice: str = "en-us"):
        self.sr, self.voice = sr, voice

    def align(self, audio: np.ndarray, transcript: str) -> Alignment:
        ref_audio, ref_alignment = synthesize_espeak(
            transcript,
            voice=self.voice,
            pitch_range=(50, 50),
            gain_jitter_db=0.0,
            sample_rate=self.sr,
        )
        out = ReferenceDTWAligner(ref_audio, ref_alignment, self.sr).align(audio)
        return out.model_copy(update={"aligner": self.name})
