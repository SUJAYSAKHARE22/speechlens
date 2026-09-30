"""Command line interface: ``speechlens <command>``."""

from __future__ import annotations

import argparse
import logging
import sys

from speechlens.config import load_config
from speechlens.dataset.builder import DatasetBuilder
from speechlens.dataset.fixtures import make_synthetic_base
from speechlens.dataset.verify import verify_dataset


def _cmd_fixtures(args: argparse.Namespace) -> int:
    path = make_synthetic_base(args.out / args.name, seed=args.seed)
    print(f"synthetic dev base written to {path}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="speechlens", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-fixtures", help="Generate a synthetic development base recording.")
    p.add_argument("--out", type=_path, default="data/dev_base")
    p.add_argument("--name", default="gettysburg_synth")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_cmd_fixtures)

    p = sub.add_parser("build-dataset", help="Build the contrastive dataset from base recordings.")
    p.add_argument("--base", type=_path, required=True, help="Folder of base-recording folders.")
    p.add_argument("--out", type=_path, required=True)
    p.add_argument("--config", type=_path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("verify", help="Verify dataset integrity (hashes, intervals, alignment).")
    p.add_argument("dataset", type=_path)
    p.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    return int(args.func(args))


def _path(s: str):
    from pathlib import Path

    return Path(s)


if __name__ == "__main__":
    sys.exit(main())
