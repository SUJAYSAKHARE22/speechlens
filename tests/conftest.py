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
