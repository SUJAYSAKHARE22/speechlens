import numpy as np

from speechlens.audio import fade_edges, load_audio, pcm16_sha256, save_audio
from speechlens.text import boundary_kind, tokenize


def test_tokenize_attaches_punctuation_and_handles_dashes():
    toks = tokenize('We can not dedicate\u2014we can not consecrate. "Now," he said.')
    assert [t.text for t in toks][:5] == ["We", "can", "not", "dedicate", "we"]
    assert boundary_kind(toks[3].punct_after) == "clause"
    assert boundary_kind(toks[7].punct_after) == "sentence"  # "consecrate."
    assert boundary_kind(toks[0].punct_after) == "none"
    assert toks[-1].text == "said" and boundary_kind(toks[-1].punct_after) == "sentence"


def test_tokenize_keeps_hyphenated_words():
    assert [t.text for t in tokenize("a battle-field of war")] == ["a", "battle-field", "of", "war"]


def test_save_load_roundtrip_is_hash_stable(tmp_path):
    rng = np.random.default_rng(0)
    x = np.clip(rng.standard_normal(16000) * 0.1, -1, 1)
    for ext in ("wav", "flac"):
        save_audio(tmp_path / f"a.{ext}", x, 16000)
        y = load_audio(tmp_path / f"a.{ext}", 16000)
        assert pcm16_sha256(y) == pcm16_sha256(x)


def test_load_resamples(tmp_path):
    x = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000) * 0.5
    save_audio(tmp_path / "a.wav", x, 8000)
    assert len(load_audio(tmp_path / "a.wav", 16000)) == 16000


def test_fade_edges_zero_at_ends_and_length_preserved():
    y = fade_edges(np.ones(1000), 50)
    assert len(y) == 1000 and y[0] == 0 and y[-1] == 0 and y[500] == 1


def test_default_yaml_matches_builtin_defaults():
    from pathlib import Path

    from speechlens.config import Config, load_config

    assert load_config(Path(__file__).parents[1] / "configs" / "default.yaml") == Config()
