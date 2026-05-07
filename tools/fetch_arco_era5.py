#!/usr/bin/env python3
"""Fetch a single ERA5 initialization from the public ARCO Zarr store.

This writes a local NetCDF file matching the weathergraph-rs / Keisler 2022
input contract:

- variables:
  specific_humidity, temperature, u_component_of_wind, v_component_of_wind,
  vertical_velocity, geopotential
- dims: (time=1, level=13, latitude=181, longitude=360)
- latitude descending 90 -> -90
- longitude ascending 0 -> 359

The implementation avoids the slower xarray+dask path by downloading the exact
compressed hourly Zarr chunk for each required variable over HTTPS, then
decompressing it locally.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import pathlib
import urllib.request

import numpy as np

ARCO_BASE = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/"
    "ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
REQUIRED_VARS = (
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
)
REQUIRED_LEVELS = np.array(
    [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    dtype=np.int64,
)
REQUIRED_LEVEL_INDICES = [8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 23, 25, 27]
TARGET_LAT = np.arange(90, -90.1, -1.0, dtype=np.float32)
TARGET_LON = np.arange(0, 360, 1.0, dtype=np.float32)
ARCO_EPOCH = dt.datetime(1900, 1, 1, tzinfo=dt.timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True, help="Initialization time, e.g. 2020-01-01T00:00:00Z")
    parser.add_argument("--out", required=True, help="Output NetCDF path")
    parser.add_argument("--workers", type=int, default=3, help="Parallel variable downloads")
    return parser.parse_args()


def load_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def load_array(url: str, dtype: np.dtype[np.generic]) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=300) as response:
        return np.frombuffer(response.read(), dtype=dtype)


def parse_init_timestamp(value: str) -> dt.datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    else:
        parsed = parsed.astimezone(dt.timezone.utc)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("ARCO fetch requires an hourly timestamp")
    return parsed


def hour_index(init_time: dt.datetime) -> int:
    delta = init_time - ARCO_EPOCH
    return int(delta.total_seconds() // 3600)


def fetch_variable(name: str, time_idx: int, selected_levels: list[int]) -> np.ndarray:
    from numcodecs.blosc import Blosc

    meta = load_json(f"{ARCO_BASE}/{name}/.zarray")
    chunk_shape = tuple(int(dimension) for dimension in meta["chunks"])
    dtype = np.dtype(str(meta["dtype"]))
    chunk_url = f"{ARCO_BASE}/{name}/{time_idx}.0.0.0"

    with urllib.request.urlopen(chunk_url, timeout=300) as response:
        compressed = response.read()

    decoded = Blosc().decode(compressed)
    full = np.frombuffer(decoded, dtype=dtype).reshape(chunk_shape)
    return np.asarray(full[0, selected_levels, ::4, ::4], dtype=np.float32)


def build_dataset(init_time: dt.datetime, workers: int):
    import xarray as xr

    selected_levels = REQUIRED_LEVEL_INDICES
    idx = hour_index(init_time)
    coords = {
        "time": np.array([np.datetime64(init_time.replace(tzinfo=None))]),
        "level": REQUIRED_LEVELS,
        "latitude": TARGET_LAT,
        "longitude": TARGET_LON,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_variable, name, idx, selected_levels): name
            for name in REQUIRED_VARS
        }
        data_vars = {}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            data = future.result()
            data_vars[name] = (("time", "level", "latitude", "longitude"), data[np.newaxis, ...])

    ordered = {name: data_vars[name] for name in REQUIRED_VARS}
    dataset = xr.Dataset(data_vars=ordered, coords=coords)
    return dataset.astype(np.float32)


def main() -> int:
    args = parse_args()
    init_time = parse_init_timestamp(args.init)
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(init_time, args.workers)
    dataset.to_netcdf(output, engine="netcdf4")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
