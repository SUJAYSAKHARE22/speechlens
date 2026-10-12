"""Plot an ideal recording against a botched one with ground-truth flaw regions overlaid.

Usage: python scripts/plot_ground_truth.py [dataset_root] [base_id] [out.png]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from speechlens.acoustics.pitch import estimate_f0_range, f0_contour
from speechlens.audio import load_audio
from speechlens.schema import VariantLabel

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/dev_corpus/datasets/B")
base = sys.argv[2] if len(sys.argv) > 2 else "B"
out = sys.argv[3] if len(sys.argv) > 3 else "docs/ground_truth_check.png"
SR = 16_000
COL = {"rush": "#e4572e", "drag": "#f3a712", "misplaced_pause": "#29335c", "missing_pause": "#6a4c93",
       "overlong_pause": "#1982c4", "monotone": "#5aa02c", "trail_off": "#ff595e",
       "volume_collapse": "#5c4d7d", "mumble": "#8d6a9f", "uptalk": "#2a9d8f"}  # fmt: skip

ideal = VariantLabel.load(root / base / "labels" / "L0_ideal.json")
botched = VariantLabel.load(root / base / "labels" / "L5_botched.json")
xi, xb = (load_audio(root / lb.audio_file, SR) for lb in (ideal, botched))
f0r = estimate_f0_range(xi, SR)
fig, ax = plt.subplots(4, 1, figsize=(15, 8.5))


def panel(aw, af, x, lb, title):
    aw.plot(np.arange(len(x)) / SR, x, lw=0.35, color="#333")
    aw.set_title(title, loc="left", fontsize=11, fontweight="bold")
    aw.set_ylabel("amplitude")
    t, f = f0_contour(x, SR, f0r.floor, f0r.ceiling)
    af.plot(t, 12 * np.log2(f / f0r.median), ".", ms=2, color="#1b4965")
    af.set_ylabel("F0 (st re median)")
    af.set_ylim(-9, 9)
    for fl in lb.flaws:
        for a in (aw, af):
            a.axvspan(fl.start, fl.end, color=COL[fl.type], alpha=0.25, lw=0)
        aw.text((fl.start + fl.end) / 2, 0.7, fl.type.replace("_", " "), ha="center", va="top",
                fontsize=7.5, rotation=90, color=COL[fl.type], fontweight="bold")  # fmt: skip
    for a in (aw, af):
        a.set_xlim(0, max(len(xi), len(xb)) / SR)


panel(ax[0], ax[1], xi, ideal, "L0 ideal baseline (no flaws)")
panel(
    ax[2],
    ax[3],
    xb,
    botched,
    f"L5 botched: {len(botched.flaws)} injected flaws; shaded = ground-truth labels",
)
ax[3].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig(out, dpi=110)
print("saved", out)
