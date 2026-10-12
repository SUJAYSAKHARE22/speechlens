"""Forced alignment backends: map transcript words to audio timestamps."""

from speechlens.alignment.dtw import ReferenceDTWAligner, TTSDTWAligner
from speechlens.alignment.refine import refine_boundaries
from speechlens.alignment.textgrid import alignment_from_textgrid

__all__ = ["ReferenceDTWAligner", "TTSDTWAligner", "alignment_from_textgrid", "refine_boundaries"]
