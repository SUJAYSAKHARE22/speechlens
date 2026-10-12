import numpy as np
import pytest

from speechlens.alignment import (
    TTSDTWAligner,
    alignment_from_textgrid,
    refine_boundaries,
)
from speechlens.features.extract import extract_frames
from speechlens.tts import espeak_available
from tests.conftest import SR


def _errors_ms(est, truth):
    e = [abs(a.start - b.start) for a, b in zip(est.words, truth.words, strict=True)]
    e += [abs(a.end - b.end) for a, b in zip(est.words, truth.words, strict=True)]
    return np.array(e) * 1000


@pytest.mark.parametrize("variant", ["L0_ideal", "L5_botched", "S_overlong_pause_60", "S_rush_60"])
def test_reference_dtw_recovers_exact_word_timings_across_speakers(reference, b_dataset, variant):
    p = b_dataset[variant]
    est = reference.aligner.align(p.audio)
    err = _errors_ms(est, p.alignment)
    assert len(est.words) == len(p.alignment.words)
    assert np.median(err) < 30 and np.percentile(err, 90) < 80
    assert est.diagnostics["dtw_cost_per_step"] > 0


def test_alignment_is_monotone_and_deterministic(reference, b_dataset):
    p = b_dataset["L3_mediocre"]
    a, b = reference.aligner.align(p.audio), reference.aligner.align(p.audio)
    assert a == b
    starts = [w.start for w in a.words]
    assert starts == sorted(starts)
    assert all(w.end >= w.start for w in a.words)


def test_cost_rises_when_audio_does_not_match_the_text(reference):
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(reference.audio)) * 0.05
    good = reference.aligner.align(reference.audio).diagnostics["dtw_cost_per_step"]
    bad = reference.aligner.align(noise).diagnostics["dtw_cost_per_step"]
    assert bad > good + 0.5


def test_refinement_recovers_true_pauses_from_a_jittered_alignment(b_dataset):
    """Perturb every boundary by up to +-50 ms; refinement must recover the real pauses."""
    p = b_dataset["L5_botched"]
    truth = p.alignment
    rng = np.random.default_rng(0)
    jittered = truth.model_copy(
        update={
            "words": [
                w.model_copy(update={"start": w.start + rng.uniform(-0.05, 0.05)})
                if i % 2
                else w.model_copy(update={"end": w.end + rng.uniform(-0.05, 0.05)})
                for i, w in enumerate(truth.words)
            ]
        }
    )
    jittered = jittered.model_copy(
        update={
            "words": [w.model_copy(update={"end": max(w.end, w.start)}) for w in jittered.words]
        }
    )
    frames, _ = extract_frames(p.audio, SR)
    refined = refine_boundaries(jittered, frames.times, frames.energy_db)
    assert refined.aligner.endswith("+refined")

    def pauses(al):
        return np.array([b.start - a.end for a, b in zip(al.words, al.words[1:], strict=False)])

    real = pauses(truth) > 0.3  # audible pauses
    err = np.abs(pauses(refined) - pauses(truth))[real]
    assert real.sum() >= 5 and np.median(err) < 0.06 and err.max() < 0.25
    assert [w.start for w in refined.words] == sorted(w.start for w in refined.words)


def test_refinement_does_not_clip_words_inside_a_quiet_passage(b_dataset):
    """Quiet speech must not be mistaken for silence (regression: global threshold)."""
    p = b_dataset["S_volume_collapse_60"]
    frames, _ = extract_frames(p.audio, SR)
    refined = refine_boundaries(p.alignment, frames.times, frames.energy_db)
    f = p.gt[0]
    quiet = [
        (a, b)
        for a, b in zip(p.alignment.words, refined.words, strict=True)
        if f[1] - 1e-3 <= a.start and a.end <= f[2] + 1e-3
    ]
    assert len(quiet) >= 3
    for truth, got in quiet:
        assert abs((got.end - got.start) - (truth.end - truth.start)) < 0.06


@pytest.mark.skipif(not espeak_available(), reason="espeak-ng not installed")
def test_tts_dtw_aligns_without_any_reference_recording(reference, b_dataset):
    p = b_dataset["L0_ideal"]
    est = TTSDTWAligner(SR).align(p.audio, p.transcript)
    err = _errors_ms(est, p.alignment)
    assert np.median(err) < 120  # coarser than reference-DTW, still word-accurate
    assert est.aligner == "dtw-tts"


def _textgrid(alignment) -> str:
    items = []
    t = 0.0
    for w in alignment.words:
        if w.start > t:
            items.append((t, w.start, ""))
        items.append((w.start, w.end, w.text.lower()))
        t = w.end
    body = "\n".join(
        f"        intervals [{k}]:\n            xmin = {a}\n            xmax = {b}\n"
        f'            text = "{x}"'
        for k, (a, b, x) in enumerate(items, 1)
    )
    return (
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\nitem []:\n'
        '    item [1]:\n        class = "IntervalTier"\n        name = "words"\n'
        f"        intervals: size = {len(items)}\n{body}\n"
        '    item [2]:\n        class = "IntervalTier"\n        name = "phones"\n'
        "        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n"
        '            xmax = 1\n            text = "x"\n'
    )


def test_textgrid_import_roundtrip_and_mismatch(tmp_path, reference):
    tg = tmp_path / "a.TextGrid"
    tg.write_text(_textgrid(reference.alignment), encoding="utf-8")
    got = alignment_from_textgrid(tg, reference.transcript, SR)
    assert [w.text for w in got.words] == [w.text for w in reference.alignment.words]
    assert got.words[10].start == pytest.approx(reference.alignment.words[10].start)
    assert got.words[10].punct_after == reference.alignment.words[10].punct_after
    with pytest.raises(ValueError, match="word intervals"):
        alignment_from_textgrid(tg, "just three words", SR)
    with pytest.raises(ValueError, match="no tier"):
        alignment_from_textgrid(tg, reference.transcript, SR, tier="nonexistent")
