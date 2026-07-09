"""Command-line interface: ``receipts check trace.json``."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from .report import check_corpus, check_trace
from .selfeval import run as run_selfeval
from .trace import load_trace_file


def _expand(paths: list[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.json"))))
        else:
            files.append(p)
    return files


def _cmd_check(args: argparse.Namespace) -> int:
    files = _expand(args.paths)
    if not files:
        print("receipts: no trace files found", file=sys.stderr)
        return 2

    traces = [load_trace_file(f) for f in files]
    if len(traces) == 1:
        report = check_trace(traces[0])
        print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
        exit_bad = report.phantom + report.silent_fail
    else:
        corpus = check_corpus(traces)
        print(json.dumps(corpus.to_dict(), indent=2) if args.json else corpus.render())
        exit_bad = corpus.phantom + corpus.silent_fail

    # Exit non-zero when anything is phantom/silent-fail, unless --no-fail.
    if exit_bad and not args.no_fail:
        return 1
    return 0


def _cmd_selfeval(args: argparse.Namespace) -> int:
    sc = run_selfeval()
    print(json.dumps(sc.to_dict(), indent=2) if args.json else sc.render())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="receipts",
        description="Reconcile what an agent CLAIMED it did against its tool-call trace.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="check one or more trace JSON files")
    check.add_argument("paths", nargs="+", help="trace file(s) or directory")
    check.add_argument("--json", action="store_true", help="emit JSON")
    check.add_argument(
        "--no-fail",
        action="store_true",
        help="always exit 0 (default exits 1 on phantom/silent-fail)",
    )
    check.set_defaults(func=_cmd_check)

    ev = sub.add_parser("selfeval", help="report the checker's own accuracy")
    ev.add_argument("--json", action="store_true", help="emit JSON")
    ev.set_defaults(func=_cmd_selfeval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
