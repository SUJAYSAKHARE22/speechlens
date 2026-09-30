import csv
import json
import shutil

import pytest

from speechlens.config import Config
from speechlens.dataset.builder import DatasetBuilder, derive_seed, load_base
from speechlens.dataset.flaws import FlawType
from speechlens.dataset.verify import verify_dataset
from speechlens.schema import VariantLabel


@pytest.fixture(scope="module")
def built(base_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("ds")
    labels = DatasetBuilder(Config()).build_all(base_dir, out)
    return out, labels


def test_structure_and_counts(built):
    out, labels = built
    assert len(labels) == 6 + len(FlawType)  # ladder + one single per flaw type
    with open(out / "index.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(labels)
    for lb in labels:
        assert (out / lb.audio_file).exists()
        assert (out / lb.base_id / "labels" / f"{lb.variant_id}.json").exists()


def test_verify_passes_and_detects_tampering(built, tmp_path):
    out, _ = built
    assert verify_dataset(out) == []
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    victim = next((copy / "synth" / "audio").glob("L3*.flac"))
    victim.write_bytes(next((copy / "synth" / "audio").glob("L1*.flac")).read_bytes())
    assert any("hash mismatch" in p for p in verify_dataset(copy))


def test_ideal_variant_is_identical_to_the_baseline(built, base_dir):
    from speechlens.audio import pcm16_sha256

    _, labels = built
    ideal = next(lb for lb in labels if lb.kind == "ideal")
    assert ideal.flaws == [] and ideal.flaw_burden == 0.0
    assert ideal.audio_sha256 == pcm16_sha256(load_base(base_dir / "synth", 16000).audio)


def test_ladder_gets_monotonically_worse(built):
    _, labels = built
    ladder = sorted(
        (lb for lb in labels if lb.kind in ("ideal", "ladder")), key=lambda lb: lb.level
    )
    counts = [len(lb.flaws) for lb in ladder]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    mean_sev = [sum(f.severity for f in lb.flaws) / len(lb.flaws) for lb in ladder[1:]]
    assert mean_sev == sorted(mean_sev)
    assert ladder[-1].flaw_burden == max(lb.flaw_burden for lb in labels)


def test_labels_roundtrip_and_flaws_are_disjoint_and_ordered(built):
    out, labels = built
    for lb in labels:
        again = VariantLabel.load(out / lb.base_id / "labels" / f"{lb.variant_id}.json")
        assert again == lb
        spans = sorted((f.start, f.end) for f in lb.flaws)
        assert all(e1 <= s2 + 1e-3 for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False))


def test_build_is_reproducible_and_seed_sensitive(base_dir, built, tmp_path):
    _, first = built
    second = DatasetBuilder(Config()).build_all(base_dir, tmp_path / "again")
    assert [lb.audio_sha256 for lb in first] == [lb.audio_sha256 for lb in second]
    cfg = Config()
    cfg.dataset.seed = 99
    third = DatasetBuilder(cfg).build_all(base_dir, tmp_path / "other")
    assert [lb.audio_sha256 for lb in first] != [lb.audio_sha256 for lb in third]


def test_seed_derivation_is_stable():
    assert derive_seed(1, "a", "b") == derive_seed(1, "a", "b")
    assert derive_seed(1, "a", "b") != derive_seed(1, "a", "c")


def test_mismatched_transcript_is_rejected(base_dir, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(base_dir / "synth", bad)
    (bad / "transcript.txt").write_text("only three words\n")
    with pytest.raises(ValueError, match="alignment has"):
        load_base(bad, 16000)


def test_label_json_is_human_readable(built):
    out, _ = built
    doc = json.loads((out / "synth" / "labels" / "L3_mediocre.json").read_text())
    assert {"flaws", "alignment", "audio_sha256", "flaw_burden"} <= doc.keys()
    assert {"type", "severity", "start", "end", "words", "params"} <= doc["flaws"][0].keys()
