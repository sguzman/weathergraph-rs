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

The Rust test at `crates/weathergraph-core/tests/parity_fixture.rs` runs automatically when the fixture exists. If the fixture directory or manifest is absent, the parity test is skipped instead of failing the workspace.
