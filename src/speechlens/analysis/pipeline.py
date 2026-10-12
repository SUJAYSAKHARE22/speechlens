"""End-to-end analysis: audio + transcript (+ reference) -> flaw regions, scores, overlay."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from speechlens.alignment import ReferenceDTWAligner, TTSDTWAligner
from speechlens.analysis.compare import (
    Deviations,
    NormModel,
    compare_to_norms,
    compare_to_reference,
    z_scores,
)
from speechlens.analysis.detect import detect_flaws
from speechlens.analysis.explain import explain_region
from speechlens.analysis.models import AnalysisResult, Series, WordReport
from speechlens.analysis.rubric import score_speech
from speechlens.config import Config
from speechlens.dataset.builder import load_base
from speechlens.features import SpeechFeatures, extract_speech_features
from speechlens.schema import Alignment
from speechlens.text import tokenize

SERIES_STRIDE = 2  # 10 ms frames -> 50 Hz series for the dashboard


@dataclass
class Reference:
    """A good ('ideal') recording of a text with its word alignment (computed lazily)."""

    id: str
    audio: np.ndarray
    sr: int
    transcript: str
    alignment: Alignment
    _features: SpeechFeatures | None = field(default=None, repr=False)
    _aligner: ReferenceDTWAligner | None = field(default=None, repr=False)

    @classmethod
    def load(cls, directory: str | Path, sr: int = 16_000) -> Reference:
        b = load_base(directory, sr)
        return cls(b.id, b.audio, sr, b.transcript, b.alignment)

    @property
    def features(self) -> SpeechFeatures:
        if self._features is None:
            self._features = extract_speech_features(self.audio, self.sr, self.alignment)
        return self._features

    @property
    def aligner(self) -> ReferenceDTWAligner:
        if self._aligner is None:
            self._aligner = ReferenceDTWAligner(self.audio, self.alignment, self.sr)
        return self._aligner


def _nullable(a: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), 2) for v in a]


def _warped_reference_series(
    par: SpeechFeatures, ref: SpeechFeatures, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the baseline contours on the participant's time axis via word-anchored warping."""
    pw, rw = par.words, ref.words
    anchors_p = [0.0, *[v for w in pw for v in (w.start, w.end)], par.alignment.duration]
    anchors_r = [0.0, *[v for w in rw for v in (w.start, w.end)], ref.alignment.duration]
    p, r = np.maximum.accumulate(anchors_p), np.maximum.accumulate(anchors_r)
    t_ref = np.interp(t, p, r)
    idx = np.clip(
        np.round(t_ref / (ref.frames.times[1] - ref.frames.times[0])).astype(int),
        0,
        len(ref.frames.times) - 1,
    )
    return ref.frames.f0_st[idx], ref.frames.energy_db[idx]


class Analyzer:
    """Stateless analysis entry point (cache references to reuse their features/aligners)."""

    def __init__(self, config: Config | None = None, norms: NormModel | None = None):
        self.cfg = config or Config()
        self.norms = norms or NormModel()

    def analyze(
        self,
        audio: np.ndarray,
        transcript: str,
        *,
        sr: int = 16_000,
        reference: Reference | None = None,
        alignment: Alignment | None = None,
        preset: str = "balanced",
    ) -> AnalysisResult:
        n_words = len(tokenize(transcript))
        warnings: list[str] = []
        if reference is not None and len(tokenize(reference.transcript)) != n_words:
            raise ValueError("transcript does not match the reference text (different word count)")

        # 1. forced alignment -------------------------------------------------------------
        if alignment is None:
            alignment = (
                reference.aligner.align(audio)
                if reference is not None
                else TTSDTWAligner(sr).align(audio, transcript)
            )
        punct = [t.punct_after for t in tokenize(transcript)]
        alignment = alignment.model_copy(
            update={
                "words": [
                    w.model_copy(update={"punct_after": p})
                    for w, p in zip(alignment.words, punct, strict=True)
                ]
            }
        )
        diag = dict(alignment.diagnostics)
        if diag.get("dtw_cost_per_step", 0.0) > 3.5:
            warnings.append(
                "High alignment cost: the audio may not match the transcript, or is very noisy. "
                "Timestamps and flaw regions may be unreliable."
            )
        ratio = diag.get("duration_ratio")
        if ratio is not None and not 0.4 <= ratio <= 2.5:
            warnings.append(
                f"Recording is {ratio:.1f}x the reference duration; check the transcript."
            )

        # 2. features ----------------------------------------------------------------------
        par = extract_speech_features(audio, sr, alignment)

        # 3. deviation -> flaws -> explanations -> scores -----------------------------------------
        dev: Deviations
        if reference is not None:
            dev = compare_to_reference(
                reference.features, par, self.cfg.analysis.detection.pitch_half_window
            )
        else:
            dev = compare_to_norms(par, self.norms, self.cfg.analysis.detection.pitch_half_window)
            warnings.append(
                "No reference supplied: scored against generic norms (lower confidence)."
            )
        tol, det = self.cfg.analysis.tolerance, self.cfg.analysis.detection
        z = z_scores(dev, tol)
        raws = detect_flaws(dev, z, par.words, det)
        regions = [explain_region(r, dev, z, par.words, tol, det, k) for k, r in enumerate(raws, 1)]
        scores = score_speech(z, preset, det)

        # 4. dashboard payload ------------------------------------------------------------------
        fr = par.frames
        sl = slice(None, None, SERIES_STRIDE)
        t = fr.times[sl]
        f0_ref = e_ref = None
        if reference is not None:
            f0r, er = _warped_reference_series(par, reference.features, t)
            f0_ref, e_ref = _nullable(f0r), [round(float(v), 2) for v in er]
        series = Series(
            t=[round(float(v), 3) for v in t],
            f0_st=_nullable(fr.f0_st[sl]),
            f0_st_ref=f0_ref,
            energy_db=[round(float(v), 2) for v in fr.energy_db[sl]],
            energy_db_ref=e_ref,
        )
        words = [
            WordReport(
                index=w.index,
                text=w.text,
                start=w.start,
                end=w.end,
                boundary=w.boundary,
                pace_log2=round(float(dev.pace[i]), 3),
                pause_after=round(float(dev.pause_obs[i]), 3),
                pause_expected=round(float(dev.pause_exp[i]), 3),
                z={k: round(float(v[i]), 2) for k, v in z.items()},
            )
            for i, w in enumerate(par.words)
        ]
        return AnalysisResult(
            mode=dev.mode, aligner=par.alignment.aligner, duration=round(len(audio) / sr, 3),
            reference_id=reference.id if reference else None, tempo_factor=dev.tempo,
            speaker={
                "f0_median_hz": round(par.stats.f0_median_hz, 1),
                "articulation_s_per_syllable": round(par.stats.articulation_s_per_syll, 3),
            },
            scores=scores, regions=regions, words=words, series=series, warnings=warnings,
            alignment_diagnostics=diag,
        )  # fmt: skip
