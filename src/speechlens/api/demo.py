"""Assemble the demo data folder used by the dashboard (references + sample recordings)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from speechlens.audio import load_audio, save_audio
from speechlens.schema import Alignment, VariantLabel

# (dataset speaker, variant id, title, description)
SAMPLES = [
    ("B", "L0_ideal", "Good reading, different speaker",
     "A flawless delivery by a different voice: should be left almost untouched."),
    ("B", "L3_mediocre", "Mediocre delivery",
     "Three injected flaws of moderate severity."),
    ("B", "L5_botched", "Botched delivery",
     "Seven injected flaws of high severity across pace, pauses, pitch, volume and clarity."),
    ("B", "S_overlong_pause_60", "Single flaw: overlong pause",
     "A two-second hesitation at a comma."),
    ("B", "S_monotone_60", "Single flaw: monotone",
     "Pitch range flattened over a long phrase."),
    ("B", "S_rush_60", "Single flaw: rushing",
     "One phrase delivered about 1.5x too fast."),
    ("B", "S_trail_off_60", "Single flaw: trailing off",
     "Energy fades away at the end of a sentence."),
]  # fmt: skip


def build_demo_data(corpus: str | Path, out: str | Path, reference: str = "A") -> Path:
    """Copy one reference and its sample participants (with ground truth) into ``out``."""
    corpus, out = Path(corpus), Path(out)
    ref_dir = out / "references" / f"gettysburg_{reference}"
    ref_dir.mkdir(parents=True, exist_ok=True)
    src = corpus / "bases" / reference
    save_audio(ref_dir / "audio.flac", load_audio(src / "audio.wav", 16_000), 16_000)
    shutil.copy(src / "transcript.txt", ref_dir / "transcript.txt")
    Alignment.load(src / "alignment.json").save(ref_dir / "alignment.json")

    for spk, vid, title, desc in SAMPLES:
        label = VariantLabel.load(corpus / "datasets" / spk / spk / "labels" / f"{vid}.json")
        d = out / "samples" / f"{spk}_{vid}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(corpus / "datasets" / spk / label.audio_file, d / "audio.flac")
        shutil.copy(corpus / "datasets" / spk / spk / "transcript.txt", d / "transcript.txt")
        meta = {
            "title": title,
            "description": desc,
            "reference_id": f"gettysburg_{reference}",
            "ground_truth": [
                {"type": f.type, "start": f.start, "end": f.end, "severity": f.severity}
                for f in label.flaws
            ],
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out
