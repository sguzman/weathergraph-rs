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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Fixture output directory")
    parser.add_argument("--data-dir", required=True, help="Directory containing upstream artifacts")
    parser.add_argument("--weights-file", required=True, help="Path to exported Rust safetensors weights")
    parser.add_argument("--dataset", required=True, help="Path to a prepared one-step xarray dataset input")
    parser.add_argument("--init", required=True, help="Forecast init timestamp, e.g. 2020-01-01T00:00:00Z")
    parser.add_argument("--tolerance", type=float, default=1.0e-4, help="Max abs error tolerance")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        import xarray as xr
        from keisler_2022.config import Config as UpstreamConfig, DataConfig
        from keisler_2022.runner import Runner as UpstreamRunner
    except ImportError as error:
        raise SystemExit(
            "This script must run inside an environment with keisler_2022, xarray, pandas, and upstream deps installed."
        ) from error

    dataset = xr.load_dataset(args.dataset)
    data_dir = pathlib.Path(args.data_dir)
    upstream_config = UpstreamConfig(
        data=DataConfig(
            weights_file=pathlib.Path(args.weights_file).name,
            normalizer_file="temporal_normalizer_rk-era5-data_zarr-era5_1979begin_2020end_03hr_6phys_181lat_360lon_13levels_blosc1comp_Corder_monolith.npz.gz",
            senders_receivers_encoder="senders_receivers_encoder.npz.gz",
            senders_receivers_processor="senders_receivers_processor.npz.gz",
            senders_receivers_decoder="senders_receivers_decoder.npz.gz",
            node_features_e="node_features_n71042_e112246_s-8416688801745003395_r-6736346125390000850.npz.gz",
            edge_features_e="edge_features_n71042_e112246_s-8416688801745003395_r-6736346125390000850.npz.gz",
            node_features_p="node_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz",
            edge_features_p="edge_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz",
            node_features_d="node_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz",
            edge_features_d="edge_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz",
            orography_landsea_file="orography_landsea.npz.gz",
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
        "weights_file": str(pathlib.Path(args.weights_file)),
        "init": args.init,
        "steps": 1,
        "tolerance": args.tolerance,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote parity fixture to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
