#!/usr/bin/env python3
"""Export upstream Haiku weights to safetensors for weathergraph-rs."""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
from safetensors.numpy import save_file


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

    flat = flatten_params(payload)
    if not flat:
        raise ValueError("no tensor-like arrays were found in the source pickle")

    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(out))
    print(f"Loaded upstream pickle type: {type(payload)!r}", file=sys.stderr)
    print(f"Exported {len(flat)} tensors to {out}", file=sys.stderr)
    return 0


def flatten_params(payload: Any) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                visit(value, path + (sanitize_key(key),))
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                visit(value, path + (str(index),))
            return

        array = maybe_array(node)
        if array is None:
            return
        if not path:
            raise ValueError("encountered tensor-like value without a stable key path")
        flat[".".join(path)] = array

    visit(payload, ())
    return flat


def maybe_array(node: Any) -> np.ndarray | None:
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        array = np.asarray(node)
    elif isinstance(node, (int, float, bool)):
        array = np.asarray(node)
    else:
        return None

    if array.dtype == np.dtype("O"):
        return None
    if array.dtype.kind == "f":
        return np.asarray(array, dtype=np.float32)
    return np.asarray(array)


def sanitize_key(key: Any) -> str:
    if isinstance(key, tuple):
        return ".".join(sanitize_key(part) for part in key)
    return str(key).replace("/", ".")


if __name__ == "__main__":
    raise SystemExit(main())
