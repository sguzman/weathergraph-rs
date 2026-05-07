#!/usr/bin/env python3
"""Export one-step reference tensors from the upstream keisler-2022 model.

This script is meant to run in a Python environment where the upstream project
and its dependencies are already available. It writes a small fixture directory
that the Rust parity test can consume.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
from safetensors.numpy import save_file


def patch_h3_compat() -> None:
    try:
        import h3
    except ImportError:
        return

    if not hasattr(h3, "get_res0_indexes") and hasattr(h3, "get_res0_cells"):
        h3.get_res0_indexes = h3.get_res0_cells
    if not hasattr(h3, "h3_to_children") and hasattr(h3, "cell_to_children"):
        h3.h3_to_children = h3.cell_to_children
    if not hasattr(h3, "h3_to_geo") and hasattr(h3, "cell_to_latlng"):
        h3.h3_to_geo = h3.cell_to_latlng


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Fixture output directory")
    parser.add_argument("--data-dir", required=True, help="Directory containing upstream artifacts")
    parser.add_argument(
        "--weights",
        "--weights-file",
        dest="rust_weights_file",
        required=True,
        help="Path to exported Rust safetensors weights",
    )
    parser.add_argument(
        "--upstream-pkl",
        help="Optional path to the upstream Haiku pickle used by the Python runner",
    )
    parser.add_argument("--dataset", required=True, help="Path to a prepared one-step xarray dataset input")
    parser.add_argument("--init", required=True, help="Forecast init timestamp, e.g. 2020-01-01T00:00:00Z")
    parser.add_argument("--tolerance", type=float, default=1.0e-4, help="Max abs error tolerance")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        patch_h3_compat()
        import pandas as pd
        import xarray as xr
        from keisler_2022.config import Config as UpstreamConfig, DataConfig
        from keisler_2022.runner import Runner as UpstreamRunner
    except ImportError as error:
        raise SystemExit(
            "This script must run inside an environment with keisler_2022, xarray, pandas, and upstream deps installed."
        ) from error

    data_dir = pathlib.Path(args.data_dir)
    dataset = xr.load_dataset(args.dataset)
    upstream_weights = pathlib.Path(args.upstream_pkl) if args.upstream_pkl else None
    rust_weights = pathlib.Path(args.rust_weights_file)
    default_data_config = DataConfig()
    upstream_weights_name = (
        upstream_weights.name if upstream_weights else default_data_config.weights_file
    )

    upstream_config = UpstreamConfig(
        data=DataConfig(
            weights_file=upstream_weights_name,
            normalizer_file=default_data_config.normalizer_file,
            senders_receivers_encoder=default_data_config.senders_receivers_encoder,
            senders_receivers_processor=default_data_config.senders_receivers_processor,
            senders_receivers_decoder=default_data_config.senders_receivers_decoder,
            node_features_e=default_data_config.node_features_e,
            edge_features_e=default_data_config.edge_features_e,
            node_features_p=default_data_config.node_features_p,
            edge_features_p=default_data_config.edge_features_p,
            node_features_d=default_data_config.node_features_d,
            edge_features_d=default_data_config.edge_features_d,
            orography_landsea_file=default_data_config.orography_landsea_file,
        )
    )

    runner = UpstreamRunner(verbose=False, config=upstream_config)
    prep = runner.prepare(dataset, n_steps=1)
    graphs = prep.graphs
    params = prep.params
    net_apply = prep.transformed.apply

    before = np.array(graphs["e"].nodes["data"], dtype=np.float32)
    all_solar = np.array(graphs["e"].nodes["all_solar"], dtype=np.float32)
    all_doy = np.array(graphs["e"].nodes["all_doy"], dtype=np.float32)
    orography = np.array(graphs["e"].nodes["orography"], dtype=np.float32)
    landsea = np.array(graphs["e"].nodes["landsea"], dtype=np.float32)
    graphs_after, _ = net_apply(params, graphs, 0)
    expected = np.array(graphs_after["e"].nodes["data"], dtype=np.float32)

    save_file(
        {
            "input_state": before,
            "solar": np.array(all_solar[..., 0], dtype=np.float32),
            "doy": np.array(all_doy[..., 0], dtype=np.float32),
            "orography": orography,
            "landsea": landsea,
            "expected_output": expected,
        },
        str(out_dir / "tensors.safetensors"),
    )

    manifest = {
        "data_dir": str(data_dir),
        "weights_file": str(rust_weights),
        "upstream_pkl": str(upstream_weights) if upstream_weights else upstream_weights_name,
        "init": args.init,
        "steps": 1,
        "tolerance": args.tolerance,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote parity fixture to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
