# Parity Fixtures

This directory is reserved for external parity fixtures:

- fake-weight MLP fixtures shared between Python and Rust
- exported one-step reference tensors
- future real-weight parity artifacts

The current workspace test suite keeps these fixtures synthetic and generated at test time so the repository does not vendor large upstream assets.

## One-Step Fixture Layout

If you export a real one-step parity fixture, place it at:

```text
tests/fixtures/parity/one_step/
├── manifest.json
└── tensors.safetensors
```

`manifest.json` fields:

- `data_dir`: absolute or repo-relative path to the upstream artifact directory
- `weights_file`: path to exported Rust `safetensors` weights
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

The Rust test in `crates/weathergraph-core/tests/parity_fixture.rs` automatically runs when this fixture is present. If the fixture directory or manifest is absent, the parity test is skipped.
