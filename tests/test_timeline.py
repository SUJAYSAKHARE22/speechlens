import numpy as np

from speechlens.dataset.timeline import Timeline
from tests.conftest import SR


def test_unmodified_render_is_bit_exact(source, alignment):
    r = Timeline.from_alignment(source, SR, alignment).render()
    assert np.array_equal(r.audio, source)


def test_segments_tile_the_source_without_overlap(source, alignment):
    tl = Timeline.from_alignment(source, SR, alignment)
    assert len(tl.segments) == 2 * len(alignment.words) + 1
    assert tl.segments[0].src[0] == 0 and tl.segments[-1].src[1] == len(source)
    for a, b in zip(tl.segments, tl.segments[1:], strict=False):
        assert a.src[1] == b.src[0]


def test_cuts_are_close_to_aligner_boundaries(source, alignment):
    tl = Timeline.from_alignment(source, SR, alignment, snap_ms=25)
    worst = max(abs(tl.word(i).src[0] / SR - w.start) for i, w in enumerate(alignment.words))
    assert worst <= 0.026


def test_output_alignment_matches_rendered_segments(source, alignment):
    tl = Timeline.from_alignment(source, SR, alignment)
    tl.gap(5).audio = np.zeros(SR)  # +1 s pause (length change)
    r = tl.render()
    out = tl.output_alignment(r)
    assert abs(out.duration - len(r.audio) / SR) < 1e-9
    assert out.words[5].start - tl.word(4).src[1] / SR > 0.9  # shifted by the inserted pause
