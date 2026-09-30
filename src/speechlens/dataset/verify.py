"""Integrity checks for a built dataset (used by ``speechlens verify`` and CI)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import soundfile as sf

from speechlens.schema import VariantLabel


def verify_dataset(root: str | Path) -> list[str]:
    """Return a list of human-readable problems; an empty list means the dataset is sound."""
    root, errors = Path(root), []
    labels = sorted(root.glob("*/labels/*.json"))
    if not labels:
        return [f"no label files under {root}"]
    for lp in labels:
        lb = VariantLabel.load(lp)
        tag = f"{lb.base_id}/{lb.variant_id}"
        audio_path = root / lb.audio_file
        if not audio_path.exists():
            errors.append(f"{tag}: missing audio {lb.audio_file}")
            continue
        pcm, sr = sf.read(str(audio_path), dtype="int16")
        if sr != lb.sample_rate:
            errors.append(f"{tag}: sample rate {sr} != label {lb.sample_rate}")
        if hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest() != lb.audio_sha256:
            errors.append(f"{tag}: audio hash mismatch")
        if abs(len(pcm) / sr - lb.duration) > 1e-3:
            errors.append(f"{tag}: duration mismatch")
        n_words = len(lb.alignment.words)
        for f in lb.flaws:
            if not (0 <= f.start < f.end <= lb.duration + 1e-3):
                errors.append(f"{tag}: flaw {f.flaw_id} interval outside audio")
            if not (f.start - 1e-3 <= f.focus_start <= f.focus_end <= f.end + 1e-3):
                errors.append(f"{tag}: flaw {f.flaw_id} focus outside label region")
            if not (0 <= f.word_start <= f.word_end < n_words):
                errors.append(f"{tag}: flaw {f.flaw_id} word range invalid")
        spans = sorted((f.start, f.end, f.flaw_id) for f in lb.flaws)
        for (_, e1, id1), (s2, _, id2) in zip(spans, spans[1:], strict=False):
            if s2 < e1 - 1e-3:
                errors.append(f"{tag}: flaws {id1} and {id2} overlap")
    return errors
