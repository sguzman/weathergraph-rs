#!/usr/bin/env python3
"""Export upstream Haiku weights to safetensors for weathergraph-rs.

This tool is intentionally conservative: it defines the expected export contract
and validates the source pickle path, but leaves exact upstream tree walking to
the parity milestone where the original weight file is available locally.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to the upstream Haiku pickle file")
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the output safetensors file that weathergraph-rs will consume",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = pathlib.Path(args.source)
    out = pathlib.Path(args.out)

    if not source.exists():
        raise FileNotFoundError(f"source pickle does not exist: {source}")

    with source.open("rb") as handle:
        payload: Any = pickle.load(handle)

    print("Loaded upstream pickle successfully.", file=sys.stderr)
    print(f"Top-level payload type: {type(payload)!r}", file=sys.stderr)
    print(f"Requested output path: {out}", file=sys.stderr)
    print(
        "Exact Haiku->safetensors flattening is deferred to the numeric parity milestone. "
        "See artifacts/README.md for the expected tensor naming contract.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

