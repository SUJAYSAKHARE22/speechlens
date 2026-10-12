"""Builds the contrastive dataset: for each base recording, an ideal reference, a
five-rung quality ladder, and isolated single-flaw variants -- all with exact
ground-truth flaw labels and word alignments.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from speechlens.acoustics.pitch import estimate_f0_range
from speechlens.audio import load_audio, ms_to_samples, pcm16_sha256, save_audio
from speechlens.config import Config, LevelSpec
from speechlens.dataset.dsp import RoomTone
from speechlens.dataset.flaws import BuildContext, FlawRequest, FlawType, inject
from speechlens.dataset.planner import RegionPlanner, word_voiced_fractions
from speechlens.dataset.timeline import Timeline
from speechlens.schema import Alignment, FlawLabel, VariantLabel
from speechlens.text import tokenize

log = logging.getLogger(__name__)

INDEX_COLUMNS = [
    "base_id",
    "variant_id",
    "kind",
    "level",
    "level_name",
    "n_flaws",
    "flaw_types",
    "flaw_burden",
    "duration",
    "audio_file",
    "label_file",
    "audio_sha256",
]


@dataclass
class BaseSpeech:
    id: str
    audio: np.ndarray
    sr: int
    transcript: str
    alignment: Alignment


def load_base(directory: str | Path, sample_rate: int) -> BaseSpeech:
    """Load ``audio.(wav|flac)``, ``transcript.txt`` and ``alignment.json`` from a folder."""
    d = Path(directory)
    audio_path = next(
        (d / f"audio.{e}" for e in ("wav", "flac") if (d / f"audio.{e}").exists()), None
    )
    if audio_path is None:
        raise FileNotFoundError(f"no audio.wav/audio.flac in {d}")
    transcript = (d / "transcript.txt").read_text(encoding="utf-8").strip()
    alignment = Alignment.load(d / "alignment.json")
    tokens = tokenize(transcript)
    if len(tokens) != len(alignment.words):
        raise ValueError(
            f"{d.name}: transcript has {len(tokens)} words but alignment has {len(alignment.words)}"
        )
    words = [
        w.model_copy(update={"punct_after": t.punct_after})
        for w, t in zip(alignment.words, tokens, strict=True)
    ]
    audio = load_audio(audio_path, sample_rate)
    alignment = alignment.model_copy(
        update={"words": words, "sample_rate": sample_rate, "duration": len(audio) / sample_rate}
    )
    return BaseSpeech(d.name, audio, sample_rate, transcript, alignment)


def derive_seed(global_seed: int, *parts: str) -> int:
    """Stable per-variant seed, independent of build order and Python hash randomisation."""
    digest = hashlib.sha256(":".join([str(global_seed), *parts]).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


class DatasetBuilder:
    def __init__(self, config: Config):
        self.cfg = config

    # -- planning ---------------------------------------------------------------------
    def _plan_level(
        self,
        base: BaseSpeech,
        spec: LevelSpec,
        rng: np.random.Generator,
        voiced: list[float],
    ) -> list[FlawRequest]:
        planner = RegionPlanner(base.alignment.words, rng, voiced)
        pool = [FlawType(t) for t in (spec.types or [f.value for f in FlawType])]
        order = rng.permutation(len(pool))
        requests: list[FlawRequest] = []
        for attempt in range(4 * spec.n_flaws + len(pool)):
            if len(requests) >= spec.n_flaws:
                break
            sev = round(float(rng.uniform(*spec.severity)), 3)
            req = planner.plan(pool[int(order[attempt % len(pool)])], sev)
            if req is not None:
                requests.append(req)
        if len(requests) < spec.n_flaws:
            log.warning(
                "%s: placed %d/%d flaws for level %s",
                base.id,
                len(requests),
                spec.n_flaws,
                spec.name,
            )
        return sorted(requests, key=lambda r: r.word_start)

    # -- rendering --------------------------------------------------------------------
    def _render_variant(
        self,
        base: BaseSpeech,
        ctx: BuildContext,
        out: Path,
        *,
        variant_id: str,
        kind: str,
        level: int | None,
        level_name: str,
        requests: list[FlawRequest],
        seed: int,
    ) -> VariantLabel:
        a = self.cfg.audio
        tl = Timeline.from_alignment(
            base.audio, base.sr, base.alignment, a.snap_window_ms, a.edge_fade_ms
        )
        rng = np.random.default_rng(seed)
        applied = [(req, inject(tl, ctx, req, rng)) for req in requests]
        rendered = tl.render()

        n_words = len(base.alignment.words)
        flaws, burden = [], 0.0
        for k, (req, ap) in enumerate(applied):
            s, e = tl.interval(rendered, ap.label_positions)
            fs, fe = tl.interval(rendered, ap.focus_positions)
            burden += req.severity * (req.word_end - req.word_start + 1) / n_words
            flaws.append(
                FlawLabel(
                    flaw_id=f"{base.id}/{variant_id}#{k}",
                    type=req.type.value,
                    severity=req.severity,
                    start=round(s, 4),
                    end=round(e, 4),
                    focus_start=round(fs, 4),
                    focus_end=round(fe, 4),
                    word_start=req.word_start,
                    word_end=req.word_end,
                    words=" ".join(
                        w.text for w in base.alignment.words[req.word_start : req.word_end + 1]
                    ),
                    params=ap.params,
                )
            )

        ext = self.cfg.dataset.audio_format
        audio_rel = f"{base.id}/audio/{variant_id}.{ext}"
        save_audio(out / audio_rel, rendered.audio, base.sr)
        label = VariantLabel(
            dataset_version=self.cfg.dataset.dataset_version,
            base_id=base.id,
            variant_id=variant_id,
            kind=kind,
            level=level,
            level_name=level_name,
            seed=seed,
            sample_rate=base.sr,
            duration=round(len(rendered.audio) / base.sr, 4),
            audio_file=audio_rel,
            audio_sha256=pcm16_sha256(rendered.audio),
            flaw_burden=round(burden, 5),
            flaws=flaws,
            alignment=tl.output_alignment(rendered),
        )
        label.save(_ensure(out / base.id / "labels" / f"{variant_id}.json"))
        return label

    # -- public API -------------------------------------------------------------------
    def build_base(self, base: BaseSpeech, out: str | Path) -> list[VariantLabel]:
        out = Path(out)
        ds = self.cfg.dataset
        ctx = BuildContext(
            sr=base.sr,
            f0=estimate_f0_range(base.audio, base.sr),
            tone=RoomTone.from_audio(base.audio, base.sr),
            context=ms_to_samples(self.cfg.audio.context_ms, base.sr),
        )
        voiced = word_voiced_fractions(base.audio, base.sr, base.alignment.words, ctx.f0)
        (out / base.id).mkdir(parents=True, exist_ok=True)
        (out / base.id / "transcript.txt").write_text(base.transcript + "\n", encoding="utf-8")

        labels: list[VariantLabel] = []
        for level, spec in sorted(ds.levels.items()):
            vid = f"L{level}_{spec.name}"
            seed = derive_seed(ds.seed, base.id, vid)
            reqs = (
                self._plan_level(base, spec, np.random.default_rng(seed), voiced)
                if spec.n_flaws
                else []
            )
            labels.append(
                self._render_variant(
                    base,
                    ctx,
                    out,
                    variant_id=vid,
                    kind="ideal" if level == 0 else "ladder",
                    level=level,
                    level_name=spec.name,
                    requests=reqs,
                    seed=seed,
                )
            )
            log.info("%s/%s: %d flaw(s)", base.id, vid, len(reqs))

        for ftype in FlawType:
            for sev in ds.single_flaw_severities:
                vid = f"S_{ftype.value}_{int(round(sev * 100)):02d}"
                seed = derive_seed(ds.seed, base.id, vid)
                req = RegionPlanner(base.alignment.words, np.random.default_rng(seed)).plan(
                    ftype, sev
                )
                if req is None:
                    log.warning("%s: no valid region for %s", base.id, vid)
                    continue
                labels.append(
                    self._render_variant(
                        base,
                        ctx,
                        out,
                        variant_id=vid,
                        kind="single",
                        level=None,
                        level_name=f"single:{ftype.value}",
                        requests=[req],
                        seed=seed,
                    )
                )
        return labels

    def build_all(self, base_root: str | Path, out: str | Path) -> list[VariantLabel]:
        out = Path(out)
        labels: list[VariantLabel] = []
        for d in sorted(p for p in Path(base_root).iterdir() if p.is_dir()):
            labels += self.build_base(load_base(d, self.cfg.audio.sample_rate), out)
        write_index(labels, out / "index.csv")
        return labels


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_index(labels: list[VariantLabel], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        for lb in labels:
            w.writerow(
                {
                    "base_id": lb.base_id,
                    "variant_id": lb.variant_id,
                    "kind": lb.kind,
                    "level": "" if lb.level is None else lb.level,
                    "level_name": lb.level_name,
                    "n_flaws": len(lb.flaws),
                    "flaw_types": "|".join(f.type for f in lb.flaws),
                    "flaw_burden": lb.flaw_burden,
                    "duration": lb.duration,
                    "audio_file": lb.audio_file,
                    "label_file": f"{lb.base_id}/labels/{lb.variant_id}.json",
                    "audio_sha256": lb.audio_sha256,
                }
            )
