# Parity Fixtures

This directory holds external parity fixtures generated from the upstream Python model. Large real fixtures are intentionally not committed.

## One-Step Fixture Layout

Place a real one-step fixture at:

```text
tests/fixtures/parity/one_step/
├── manifest.json
└── tensors.safetensors
```

`manifest.json` fields:

- `data_dir`: absolute or repo-relative path to the upstream artifact directory
- `weights_file`: path to the exported Rust-side `safetensors` checkpoint
- `upstream_pkl`: upstream Haiku pickle used by the Python reference run
- `init`: ISO8601 timestamp
- `steps`: currently `1`
- `tolerance`: max absolute error threshold

`tensors.safetensors` keys:

- `input_state`
- `solar`
- `doy`
- `orography`
- `landsea`
- `expected_output`

Generate this fixture with `tools/export_parity_fixture.py` from an environment where the upstream Python model can run.

Example:

```bash
python3 tools/export_parity_fixture.py \
  --data-dir /path/to/data \
  --weights /path/to/data/weights.safetensors \
  --dataset /path/to/data/era5_input.nc \
  --init 2020-01-01T00:00:00Z \
  --out-dir tests/fixtures/parity/one_step
```

The Rust test at `crates/weathergraph-core/tests/parity_fixture.rs` runs automatically when the fixture exists. If the fixture directory or manifest is absent, the parity test is skipped instead of failing the workspace.
