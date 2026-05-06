# weathergraph-rs

Rust-native inference scaffolding for the Keisler 2022 graph neural weather forecasting model.

This project ports the inference path of [`rkeisler/keisler-2022`](https://github.com/rkeisler/keisler-2022) from Python, JAX, Haiku, and Jraph into Rust using Candle-oriented tensor/model abstractions. The current scope is inference parity infrastructure, not Rust training.

## Status

`weathergraph-rs` currently implements the V1 inference-core foundation:

| Milestone | Status | Notes |
| --- | --- | --- |
| Artifact inspector | Implemented | Loads `.npz.gz` artifacts and prints typed shape summaries. |
| Geometry parity | Implemented | ERA5 and H3 geometry generation with exact node-count checks. |
| Graph aggregation kernel | Implemented | CPU-first gather and scatter-add aggregation with tests. |
| MLP and LayerNorm parity | Implemented | Explicit Rust/Candle tensor modules with fake-weight tests. |
| One-step scaffold | Implemented | Runner, model loading boundary, solar features, and one-step path. |
| CLI and logging | Implemented | `inspect-artifacts`, `inspect-geometry`, `inspect-weights`, and `forecast` command wiring. |
| Python/Rust numeric parity | Scaffolded | Fixture contract and weight-export boundary are documented. |
| Autoregressive rollout / NetCDF output | Deferred | Reserved for the next milestone. |

## Scope

The project is intentionally staged:

- In scope now:
  - artifact loading
  - geometry reconstruction
  - static graph loading
  - tensor aggregation primitives
  - explicit model module shapes
  - one-step runner scaffolding
  - CLI and documentation
- Deferred:
  - full parity against the original trained weights
  - multi-step rollout
  - NetCDF forecast output
  - Rust-native training

## Workspace Layout

```text
weathergraph-rs/
├── artifacts/
│   └── README.md
├── crates/
│   ├── weathergraph-cli/
│   └── weathergraph-core/
├── tests/
│   └── fixtures/
└── tools/
    └── export_weights.py
```

Core crate modules:

- `config.rs`: config structs and forecast request validation
- `error.rs`: shared error model
- `features.rs`: typed `.npz.gz` artifact loading and summaries
- `geometry.rs`: ERA5 and H3 geometry generation
- `graph.rs`: static graph loading
- `normalizer.rs`: temporal normalizer and surface feature loading
- `solar.rs`: deterministic solar and seasonal features
- `tensor.rs`: gather and scatter-add helpers
- `model.rs`: explicit MLP, LayerNorm, and GNN scaffolding
- `runner.rs`: high-level runner orchestration

## Upstream Relationship

The reference implementation lives in [`rkeisler/keisler-2022`](https://github.com/rkeisler/keisler-2022). `weathergraph-rs` preserves the same high-level model structure:

1. Encoder from ERA5 latitude/longitude fields to an H3 mesh
2. Processor graph message passing blocks
3. Decoder back to lat/lon outputs
4. Autoregressive rollout for longer forecasts

The Rust port does not read the original Python pickle weights directly. The boundary is an export tool that flattens the upstream Haiku parameter tree into `safetensors`.
That exporter now supports raw-key inspection, unresolved-key reporting, and explicit key remapping so the upstream Haiku module names can be reconciled with the Rust loader contract incrementally.

## Artifact Expectations

The Rust repo does not vendor the large upstream artifacts. Instead, place the upstream `.npz.gz` graph and normalizer files in a data directory and point the CLI at it.

Expected files:

- `senders_receivers_encoder.npz.gz`
- `senders_receivers_processor.npz.gz`
- `senders_receivers_decoder.npz.gz`
- `node_features_*.npz.gz`
- `edge_features_*.npz.gz`
- `temporal_normalizer*.npz.gz`
- `orography_landsea.npz.gz`
- exported `weights.safetensors`

See [artifacts/README.md](artifacts/README.md) for the exact export contract.

## CLI

Inspect artifact shapes:

```bash
cargo run -p weathergraph-cli -- inspect-artifacts --data-dir /path/to/upstream/data
```

Inspect geometry parity:

```bash
cargo run -p weathergraph-cli -- inspect-geometry
```

Inspect an exported checkpoint before trying parity or forecast:

```bash
cargo run -p weathergraph-cli -- inspect-weights \
  --weights /path/to/weights.safetensors \
  --input-channels 78 \
  --output-channels 78 \
  --hidden-dim 256
```

Validate a forecast request and runner wiring:

```bash
cargo run -p weathergraph-cli -- forecast \
  --data-dir /path/to/upstream/data \
  --weights /path/to/weights.safetensors \
  --init 2020-01-01T00:00:00Z \
  --steps 1 \
  --input era5 \
  --out ./forecast.nc
```

At this stage the `forecast` command validates the configuration, loads artifacts, exercises the one-step scaffold, and then returns a clear deferred-work error for rollout/output.

## Logging

All user-visible logging goes through `tracing`. By default the CLI logs at `info`. Override with `RUST_LOG`, for example:

```bash
RUST_LOG=debug cargo run -p weathergraph-cli -- inspect-artifacts --data-dir /path/to/data
```

## Development

Format, lint, and test the full workspace:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

## Testing

The codebase includes:

- unit tests for `.npz.gz` loading
- unit tests for ERA5/H3 geometry counts
- unit tests for gather/scatter tensor kernels
- unit tests for MLP and GNN shape behavior
- unit tests for weight-key inspection and alias matching
- CLI integration tests for `inspect-artifacts`, `inspect-geometry`, `inspect-weights`, and forecast validation

Parity against real upstream tensors is intentionally gated behind external fixtures and exported weights.

To run a real one-step parity check, export a fixture into `tests/fixtures/parity/one_step/` using `tools/export_parity_fixture.py`. When `manifest.json` and `tensors.safetensors` are present there, `cargo test --workspace` will automatically execute the parity comparison in addition to the synthetic unit/integration tests.

## Deferred Work

- Match the original Haiku parameter layout exactly and validate one-step numerical parity
- Add full graph-based feature assembly and real model state semantics
- Implement autoregressive rollout
- Write NetCDF forecast output
- Add recent ECMWF Open Data ingestion for real-time initialization
