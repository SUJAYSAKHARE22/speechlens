from __future__ import annotations

import numpy as np
import pytest

from speechlens.acoustics.pitch import estimate_f0_range
from speechlens.audio import load_audio
from speechlens.dataset.dsp import RoomTone
from speechlens.dataset.fixtures import make_synthetic_base
from speechlens.dataset.flaws import BuildContext
from speechlens.schema import Alignment

SR = 16_000


@pytest.fixture(scope="session")
def base_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("bases")
    make_synthetic_base(root / "synth")
    return root


@pytest.fixture(scope="session")
def source(base_dir) -> np.ndarray:
    return load_audio(base_dir / "synth" / "audio.wav", SR)


@pytest.fixture(scope="session")
def alignment(base_dir) -> Alignment:
    from speechlens.dataset.builder import load_base

    return load_base(base_dir / "synth", SR).alignment  # punctuation attached from transcript


@pytest.fixture(scope="session")
def ctx(source) -> BuildContext:
    return BuildContext(SR, estimate_f0_range(source, SR), RoomTone.from_audio(source, SR), 2400)


@pytest.fixture(scope="session")
def voiced(source, alignment, ctx) -> list[float]:
    from speechlens.dataset.planner import word_voiced_fractions

    return word_voiced_fractions(source, SR, alignment.words, ctx.f0)


@pytest.fixture(scope="session")
def reference(base_dir):
    from speechlens.analysis.pipeline import Reference

    return Reference.load(base_dir / "synth", SR)


@pytest.fixture(scope="session")
def speaker_b_dir(tmp_path_factory):
    from speechlens.dataset.fixtures import SPEAKERS

    root = tmp_path_factory.mktemp("speaker_b")
    make_synthetic_base(root / "B", speaker=SPEAKERS["B"])
    return root / "B"


@pytest.fixture(scope="session")
def b_dataset(speaker_b_dir, tmp_path_factory):
    """Speaker B's contrastive dataset (a different 'person' from the reference voice)."""
    from speechlens.config import Config
    from speechlens.dataset.builder import DatasetBuilder, load_base
    from speechlens.evaluation.run import load_dataset_participants

    out = tmp_path_factory.mktemp("b_ds")
    DatasetBuilder(Config()).build_base(load_base(speaker_b_dir, SR), out)
    return {p.id.split("/")[1]: p for p in load_dataset_participants(out, "B", SR)}


@pytest.fixture(scope="session")
def demo_dir(base_dir, b_dataset, tmp_path_factory):
    """A minimal dashboard data folder: one reference (speaker A) and one flawed sample."""
    import json
    import shutil

    from speechlens.audio import save_audio

    out = tmp_path_factory.mktemp("demo")
    shutil.copytree(base_dir / "synth", out / "references" / "ref_a")
    p = b_dataset["S_overlong_pause_60"]
    d = out / "samples" / "pause"
    d.mkdir(parents=True)
    save_audio(d / "audio.flac", p.audio, SR)
    (d / "transcript.txt").write_text(p.transcript, encoding="utf-8")
    meta = {
        "title": "Overlong pause",
        "description": "test sample",
        "reference_id": "ref_a",
        "ground_truth": [{"type": t, "start": s, "end": e, "severity": v} for t, s, e, v in p.gt],
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return out
