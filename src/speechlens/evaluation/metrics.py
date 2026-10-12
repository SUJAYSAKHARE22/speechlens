"""Temporal-grounding metrics: interval IoU matching, precision/recall, boundary error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Region = tuple[str, float, float]  # (type, start_s, end_s)


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Match:
    pred: int
    gt: int
    iou: float


def match_regions(pred: list[Region], gt: list[Region], threshold: float) -> list[Match]:
    """One-to-one greedy matching by descending IoU (type-agnostic: detection quality)."""
    cands = sorted(
        (
            Match(i, j, iou((p[1], p[2]), (g[1], g[2])))
            for i, p in enumerate(pred)
            for j, g in enumerate(gt)
        ),
        key=lambda m: -m.iou,
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    out: list[Match] = []
    for m in cands:
        if m.iou < threshold:
            break
        if m.pred in used_p or m.gt in used_g:
            continue
        used_p.add(m.pred)
        used_g.add(m.gt)
        out.append(m)
    return out


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


def median_ms(values: list[float]) -> float | None:
    return round(float(np.median(values)) * 1000, 1) if values else None
