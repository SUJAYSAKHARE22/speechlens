"""Command line interface: ``speechlens <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from speechlens.config import load_config
from speechlens.dataset.builder import DatasetBuilder, load_base
from speechlens.dataset.fixtures import SPEAKERS, make_synthetic_base
from speechlens.dataset.verify import verify_dataset


def _path(s: str) -> Path:
    return Path(s)


def _cmd_fixtures(args: argparse.Namespace) -> int:
    path = make_synthetic_base(args.out / args.name, seed=args.seed)
    print(f"synthetic dev base written to {path}")
    return 0


def _cmd_corpus(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    for name, sp in SPEAKERS.items():
        make_synthetic_base(args.out / "bases" / name, speaker=sp)
        base = load_base(args.out / "bases" / name, cfg.audio.sample_rate)
        DatasetBuilder(cfg).build_base(base, args.out / "datasets" / name)
        print(f"speaker {name}: {sp.voice}, {sp.speed} wpm -> base + dataset written")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.dataset.seed = args.seed
    labels = DatasetBuilder(cfg).build_all(args.base, args.out)
    print(f"built {len(labels)} variants -> {args.out}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    problems = verify_dataset(args.dataset)
    for p in problems:
        print("PROBLEM:", p)
    print("dataset OK" if not problems else f"{len(problems)} problem(s) found")
    return 1 if problems else 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from speechlens.analysis.pipeline import Analyzer, Reference
    from speechlens.audio import load_audio

    cfg = load_config(args.config)
    sr = cfg.audio.sample_rate
    ref = Reference.load(args.reference, sr) if args.reference else None
    transcript = (
        args.transcript.read_text(encoding="utf-8").strip() if args.transcript else ref.transcript
    )
    result = Analyzer(cfg).analyze(
        load_audio(args.audio, sr), transcript, sr=sr, reference=ref, preset=args.preset
    )
    text = result.model_dump_json(indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(f"overall {result.scores.overall}  categories {result.scores.categories}")
    for r in result.regions:
        print(f"  [{r.start:6.2f}-{r.end:6.2f}s] {r.headline}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from speechlens.evaluation.run import evaluate_corpus, save_report

    cfg = load_config(args.config)
    res = evaluate_corpus(
        args.corpus, args.reference, args.participants, cfg, not args.no_calibrate, args.oracle
    )
    for spk, report in res.items():
        tag = f"{args.name}_{spk}" + ("_oracle" if args.oracle else "")
        save_report(report, args.out / tag, f"Speaker {spk} vs reference {args.reference}")
        d = report["detection"]["iou>=0.3"]
        print(
            f"speaker {spk}: P {d['precision']} R {d['recall']} F1 {d['f1']} "
            f"IoU {report['matched_mean_iou']}"
        )
    return 0


def _cmd_evaluate_human(args: argparse.Namespace) -> int:
    from speechlens.analysis.pipeline import Analyzer, Reference
    from speechlens.evaluation.run import evaluate, load_human_annotations, save_report

    cfg = load_config(args.config)
    ref = Reference.load(args.reference, cfg.audio.sample_rate)
    parts = load_human_annotations(args.annotations, ref, cfg.audio.sample_rate)
    report = evaluate(Analyzer(cfg), ref, parts)
    save_report(report, args.out, f"Human-annotated recordings vs {ref.id}")
    d = report["detection"]["iou>=0.3"]
    print(f"{len(parts)} recordings: P {d['precision']} R {d['recall']} F1 {d['f1']}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from speechlens.analysis.calibration import calibrate_tolerance, good_deviations
    from speechlens.analysis.pipeline import Reference

    cfg = load_config(args.config)
    ref = Reference.load(args.reference, cfg.audio.sample_rate)
    tol = calibrate_tolerance(good_deviations(ref, args.good), cfg.analysis.tolerance)
    args.out.write_text(
        yaml.safe_dump({"analysis": {"tolerance": tol.model_dump()}}, sort_keys=False),
        encoding="utf-8",
    )
    print(f"calibrated tolerance written to {args.out}:\n{json.dumps(tol.model_dump(), indent=1)}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from speechlens.api.demo import build_demo_data

    out = build_demo_data(args.corpus, args.out, args.reference)
    print(f"demo data written to {out}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from speechlens.api.app import create_app

    uvicorn.run(create_app(args.data, load_config(args.config)), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="speechlens", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-fixtures", help="Generate one synthetic development base recording.")
    p.add_argument("--out", type=_path, default="data/dev_base")
    p.add_argument("--name", default="gettysburg_synth")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_cmd_fixtures)

    p = sub.add_parser("make-dev-corpus", help="Synthesise 3 simulated speakers + their datasets.")
    p.add_argument("--out", type=_path, default="data/dev_corpus")
    p.add_argument("--config", type=_path, default=None)
    p.set_defaults(func=_cmd_corpus)

    p = sub.add_parser("build-dataset", help="Build the contrastive dataset from base recordings.")
    p.add_argument("--base", type=_path, required=True, help="Folder of base-recording folders.")
    p.add_argument("--out", type=_path, required=True)
    p.add_argument("--config", type=_path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("verify", help="Verify dataset integrity (hashes, intervals, alignment).")
    p.add_argument("dataset", type=_path)
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("analyze", help="Analyze one recording against a reference (or norms).")
    p.add_argument("audio", type=_path)
    p.add_argument("--reference", type=_path, default=None, help="Reference base folder.")
    p.add_argument("--transcript", type=_path, default=None)
    p.add_argument("--preset", default="balanced")
    p.add_argument("--config", type=_path, default=None)
    p.add_argument("--out", type=_path, default=None, help="Write the full JSON result here.")
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("evaluate", help="Evaluate flaw grounding & scoring on a corpus.")
    p.add_argument("--corpus", type=_path, default="data/dev_corpus")
    p.add_argument("--reference", default="A")
    p.add_argument("--participants", nargs="+", default=["B", "C"])
    p.add_argument("--oracle", action="store_true", help="Use ground-truth alignments.")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--name", default="dev_synthetic")
    p.add_argument("--config", type=_path, default=None)
    p.add_argument("--out", type=_path, default="docs/evaluation")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("evaluate-human", help="Evaluate on human takes with annotated flaw spans.")
    p.add_argument("--reference", type=_path, required=True, help="Reference base folder.")
    p.add_argument("--annotations", type=_path, required=True)
    p.add_argument("--config", type=_path, default=None)
    p.add_argument("--out", type=_path, default="docs/evaluation/human")
    p.set_defaults(func=_cmd_evaluate_human)

    p = sub.add_parser("calibrate", help="Learn natural tolerance from good recordings.")
    p.add_argument("--reference", type=_path, required=True)
    p.add_argument("--good", type=_path, nargs="+", required=True)
    p.add_argument("--config", type=_path, default=None)
    p.add_argument("--out", type=_path, default="tolerance.yaml")
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("make-demo", help="Assemble the dashboard's demo data from a dev corpus.")
    p.add_argument("--corpus", type=_path, default="data/dev_corpus")
    p.add_argument("--out", type=_path, default="data/demo")
    p.add_argument("--reference", default="A")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("serve", help="Run the dashboard (FastAPI).")
    p.add_argument("--data", type=_path, default="data/demo")
    p.add_argument("--config", type=_path, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
