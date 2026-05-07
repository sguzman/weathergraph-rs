#!/usr/bin/env python3
"""Export upstream Haiku weights to safetensors for weathergraph-rs.

The exporter supports three workflows:

1. Dump the discovered raw parameter keys for inspection.
2. Apply an explicit JSON mapping from raw keys to Rust loader keys.
3. Apply a small heuristic alias pass for the known GNN update function names.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import re
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to the upstream Haiku pickle file")
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the output safetensors file that weathergraph-rs will consume",
    )
    parser.add_argument(
        "--mapping-file",
        help="Optional JSON file mapping raw flattened Haiku keys to Rust loader keys",
    )
    parser.add_argument(
        "--dump-keys",
        action="store_true",
        help="Print the discovered flattened raw keys to stderr before export",
    )
    parser.add_argument(
        "--no-auto-alias",
        action="store_true",
        help="Disable heuristic renaming of common upstream module/function names",
    )
    parser.add_argument(
        "--emit-unmapped",
        help="Optional path to write raw flattened keys that still need manual mapping after aliasing",
    )
    parser.add_argument(
        "--emit-mapping-template",
        help="Optional path to write a JSON mapping skeleton for unresolved raw keys",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip writing safetensors and only report remapped/unmapped key coverage",
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

    flat_raw = flatten_params(payload)
    if not flat_raw:
        raise ValueError("no tensor-like arrays were found in the source pickle")

    if args.dump_keys:
        print("Discovered flattened parameter keys:", file=sys.stderr)
        for key in sorted(flat_raw):
            print(f"  {key}", file=sys.stderr)

    mapping = load_mapping(args.mapping_file)
    flat_mapped, unmapped = remap_keys(flat_raw, mapping, auto_alias=not args.no_auto_alias)
    ensure_unique_keys(flat_mapped)

    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        try:
            from safetensors.numpy import save_file
        except ImportError as error:
            raise SystemExit(
                "Writing safetensors requires the Python `safetensors` package. "
                "Re-run with --dry-run to inspect key mapping without it."
            ) from error

        save_file(flat_mapped, str(out))
    if args.emit_unmapped:
        emit_unmapped(pathlib.Path(args.emit_unmapped), unmapped)
    if args.emit_mapping_template:
        emit_mapping_template(pathlib.Path(args.emit_mapping_template), unmapped)
    print(f"Loaded upstream pickle type: {type(payload)!r}", file=sys.stderr)
    if args.dry_run:
        print(
            f"Dry run mapped {len(flat_mapped)} tensors; no safetensors file was written",
            file=sys.stderr,
        )
    else:
        print(f"Exported {len(flat_mapped)} tensors to {out}", file=sys.stderr)
    if unmapped:
        print(
            f"{len(unmapped)} raw keys still require manual review or mapping",
            file=sys.stderr,
        )
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


def load_mapping(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping_path = pathlib.Path(path)
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise ValueError("mapping file must contain a JSON object of raw_key -> rust_key")
    return {str(key): str(value) for key, value in mapping.items()}


def remap_keys(
    tensors: dict[str, np.ndarray],
    explicit_mapping: dict[str, str],
    *,
    auto_alias: bool,
) -> tuple[dict[str, np.ndarray], list[str]]:
    remapped: dict[str, np.ndarray] = {}
    unmapped: list[str] = []
    for raw_key, value in tensors.items():
        mapped_key = explicit_mapping.get(raw_key)
        if mapped_key is None and auto_alias:
            mapped_key = alias_key(raw_key)
        if mapped_key is None:
            mapped_key = raw_key
            unmapped.append(raw_key)
        remapped[mapped_key] = adapt_tensor(mapped_key, value)
    return remapped, sorted(unmapped)


def adapt_tensor(mapped_key: str, value: np.ndarray) -> np.ndarray:
    if is_linear_weight_key(mapped_key) and value.ndim == 2:
        return np.asarray(value, dtype=np.float32).T
    return np.asarray(value, dtype=np.float32)


def is_linear_weight_key(mapped_key: str) -> bool:
    if mapped_key.endswith(".layer_norm.weight"):
        return False
    if mapped_key.endswith(".weight") and ".layers." in mapped_key:
        return True
    return mapped_key in {"input_projection.weight", "output_projection.weight"}


def alias_key(raw_key: str) -> str | None:
    normalized_key = normalize_flat_key(raw_key)
    module_prefix = infer_module_prefix(normalized_key)
    if module_prefix is None:
        return None

    if (
        ".layer_norm." in normalized_key
        or normalized_key.endswith(".scale")
        or normalized_key.endswith(".offset")
    ):
        return map_layer_norm_key(normalized_key, module_prefix)

    if normalized_key.endswith(".w") or normalized_key.endswith(".b"):
        return map_linear_key(normalized_key, module_prefix)

    return None


def normalize_flat_key(raw_key: str) -> str:
    normalized = raw_key.replace("/~/", ".").replace("/", ".")
    normalized = normalized.replace("._", ".")
    normalized = normalized.replace("~.", "")
    normalized = normalized.replace(".~", "")
    normalized = normalized.replace("..", ".")
    return normalized.strip(".")


def infer_module_prefix(raw_key: str) -> str | None:
    module_map = {
        "edge_update_fn_encoder": "encoder_edge_mlp",
        "node_update_fn_encoder": "encoder_node_mlp",
        "edge_update_fn_processor": "processor_edge_mlp",
        "node_update_fn_processor": "processor_node_mlp",
        "edge_update_fn_decoder": "decoder_edge_mlp",
        "node_update_fn_decoder": "decoder_node_mlp",
    }

    for needle, prefix in module_map.items():
        if needle in raw_key:
            return prefix

    if "net_edges" in raw_key:
        return "processor_edge_init_mlp"

    return None


def map_linear_key(raw_key: str, module_prefix: str) -> str:
    match = re.search(r"\.linear(?:_(\d+))?\.(w|b)$", raw_key)
    if match is None:
        raise ValueError(f"unable to infer linear layer index from key: {raw_key}")
    layer_idx = int(match.group(1) or 0)
    suffix = "weight" if raw_key.endswith(".w") else "bias"
    return f"{module_prefix}.layers.{layer_idx}.{suffix}"


def map_layer_norm_key(raw_key: str, module_prefix: str) -> str:
    if raw_key.endswith(".scale"):
        suffix = "weight"
    elif raw_key.endswith(".offset"):
        suffix = "bias"
    elif raw_key.endswith(".weight"):
        suffix = "weight"
    else:
        suffix = "bias"
    return f"{module_prefix}.layer_norm.{suffix}"


def ensure_unique_keys(tensors: dict[str, np.ndarray]) -> None:
    seen: set[str] = set()
    for key in tensors:
        if key in seen:
            raise ValueError(f"duplicate mapped tensor key after remap: {key}")
        seen.add(key)


def emit_unmapped(path: pathlib.Path, unmapped: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"unmapped_raw_keys": unmapped}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def emit_mapping_template(path: pathlib.Path, unmapped: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = {key: "" for key in unmapped}
    path.write_text(json.dumps(template, indent=2, sort_keys=True), encoding="utf-8")


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
