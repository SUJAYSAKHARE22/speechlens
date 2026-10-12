import numpy as np

from speechlens.features import extract_speech_features
from speechlens.text import count_syllables
from tests.conftest import SR


def test_pitch_is_speaker_agnostic(reference, b_dataset):
    """Two speakers with different absolute pitch have comparable semitone ranges."""
    a = reference.features
    b = extract_speech_features(b_dataset["L0_ideal"].audio, SR, b_dataset["L0_ideal"].alignment)
    assert abs(a.stats.f0_median_hz - b.stats.f0_median_hz) > 5  # genuinely different voices
    pa = np.nanpercentile(a.frames.f0_st, [5, 95])
    pb = np.nanpercentile(b.frames.f0_st, [5, 95])
    assert np.all(np.abs(pa - pb) < 1.5)
    assert abs(np.nanmedian(a.frames.f0_st)) < 0.7  # re the speaker's own median


def test_energy_is_invariant_to_recording_gain(reference):
    quiet = extract_speech_features(reference.audio * 0.25, SR, reference.alignment)
    loud = reference.features
    diff = [abs(p.energy_db - q.energy_db) for p, q in zip(quiet.words, loud.words, strict=True)]
    assert max(diff) < 0.2


def test_word_features_are_consistent(reference):
    words = reference.features.words
    assert len(words) == len(reference.alignment.words)
    for w1, w2 in zip(words, words[1:], strict=False):
        assert w1.pause_after == w2.pause_before
        assert w1.end <= w2.start + 1e-9
    assert all(w.n_syll >= 1 and w.duration > 0 for w in words)
    sentence_final = [w for w in words if w.boundary == "sentence"]
    assert sentence_final and np.nanmean([w.f0_tail_st for w in sentence_final]) < 0  # falls


def test_frame_arrays_share_one_time_grid(reference):
    f = reference.features.frames
    n = len(f.times)
    assert len(f.f0_st) == len(f.energy_db) == len(f.hf_pow) == len(f.hnr_db) == len(f.mfcc) == n
    assert f.mfcc.shape[1] == 13 and np.allclose(np.diff(f.times), 0.01, atol=1e-6)


def test_syllable_counter():
    assert [
        count_syllables(w) for w in ["a", "nation", "fathers", "consecrate", "the", "battle-field"]
    ] == [1, 2, 2, 3, 1, 3]
