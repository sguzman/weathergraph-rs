# Artifact Contract

`weathergraph-rs` keeps large upstream assets out of the repository. The Rust runtime consumes local artifacts for:

- graph connectivity
- node and edge features
- temporal normalization
- surface fields
- exported model weights
- local ERA5-style initialization input

## Upstream Source

Reference layout:

- <https://github.com/rkeisler/keisler-2022>

The upstream Python project stores its artifacts under `src/keisler_2022/data/`.

## Required Local Files

Expected data directory contents:

```text
data/
├── edge_features_*.npz.gz
├── node_features_*.npz.gz
├── orography_landsea.npz.gz
├── senders_receivers_decoder.npz.gz
├── senders_receivers_encoder.npz.gz
├── senders_receivers_processor.npz.gz
├── temporal_normalizer*.npz.gz
├── era5_input.nc
└── weights.safetensors
```

`era5_input.nc` must provide these variables on dimensions `(time=1, level=13, latitude=181, longitude=360)`:

- `specific_humidity`
- `temperature`
- `u_component_of_wind`
- `v_component_of_wind`
- `vertical_velocity`
- `geopotential`

You can generate that file from the public ARCO dataset with:

```bash
source /tmp/weathergraph-venv/bin/activate
python tools/fetch_arco_era5.py \
  --init 2020-01-01T00:00:00Z \
  --out /path/to/data/era5_input.nc
```

## Weight Export Contract

Use `tools/export_weights.py` against the upstream `.pkl` weight file. The exporter:

1. loads the upstream Haiku pickle
2. flattens the parameter tree
3. applies optional mapping and alias normalization
4. writes a single `safetensors` file
5. transposes upstream linear weights from Haiku `[in, out]` storage into the Rust `[out, in]` contract
6. preserves `f32` dtype

Recommended workflow:

1. Inspect raw flattened keys and generate unresolved-key reports:

```bash
python3 tools/export_weights.py \
  --source /path/to/upstream.pkl \
  --out /tmp/weights.safetensors \
  --dump-keys \
  --emit-unmapped /tmp/unmapped.json \
  --emit-mapping-template /tmp/mapping-template.json \
  --dry-run
```

2. If the built-in aliases are insufficient, fill in a mapping file. Start from either:

- the generated mapping template
- `tools/weight_mapping.example.json`

3. Export the final checkpoint. For the public upstream checkpoint in `rkeisler/keisler-2022`, the repo already includes the completed mapping file:

- `tools/weight_mapping.keisler_2022.json`

Use it directly unless you are exporting a different checkpoint layout.

4. Export the final checkpoint:

```bash
python3 tools/export_weights.py \
  --source /path/to/upstream.pkl \
  --out /path/to/data/weights.safetensors \
  --mapping-file tools/weight_mapping.keisler_2022.json
```

5. Validate it strictly before parity or forecast:

```bash
cargo run -p weathergraph-cli -- inspect-weights \
  --weights /path/to/data/weights.safetensors \
  --json \
  --strict \
  --input-channels 78 \
  --output-channels 78 \
  --hidden-dim 256
```

`inspect-weights --json --strict` is the checkpoint gate. Missing required tensors, shape mismatches, or dtype mismatches must be treated as export-contract failures.

## Canonical Rust Weight Keys

The Rust loader expects canonical names in this form:

- `encoder_edge_mlp.layers.0.weight`
- `encoder_edge_mlp.layers.0.bias`
- `encoder_edge_mlp.layers.1.weight`
- `encoder_edge_mlp.layers.1.bias`
- `encoder_edge_mlp.layers.2.weight`
- `encoder_edge_mlp.layers.2.bias`
- `encoder_edge_mlp.layer_norm.weight`
- `encoder_edge_mlp.layer_norm.bias`

The same layered structure applies to:

- `encoder_node_mlp`
- `processor_edge_init_mlp`
- `processor_edge_mlps.{block_index}`
- `processor_node_mlps.{block_index}`
- `decoder_edge_mlp`
- `decoder_node_mlp`

Optional projection tensors:

- `input_projection.weight`
- `input_projection.bias`
- `output_projection.weight`
- `output_projection.bias`

Alias handling in the Rust loader also accepts common alternates such as:

- `w` / `b`
- `layer_norm.scale` / `layer_norm.offset`
- selected upstream update-function prefixes

The recommended workflow is still to export canonical names and use alias support only as a compatibility bridge.

## One-Step Parity Fixture

Use `tools/export_parity_fixture.py` from an upstream Python environment to write:

```text
tests/fixtures/parity/one_step/
├── manifest.json
└── tensors.safetensors
```

That fixture feeds the Rust parity test in `crates/weathergraph-core/tests/parity_fixture.rs`.

Expected tensor keys:

- `input_state` - normalized model-space node state, not raw physical units
- `solar`
- `doy`
- `orography`
- `landsea`
- `expected_output` - normalized model-space node state after one step

The manifest supplies the fixture tolerance and the data/weight paths used to generate the reference output.

Recommended real-artifact command:

```bash
python3 tools/export_parity_fixture.py \
  --data-dir /path/to/data \
  --weights /path/to/data/weights.safetensors \
  --dataset /path/to/data/era5_input.nc \
  --init 2020-01-01T00:00:00Z \
  --out-dir tests/fixtures/parity/one_step
```

## Notes

- The Rust side does not read Python pickle directly.
- Large upstream artifacts should not be committed into this repository.
- Tiny synthetic fixtures are fine for tests.
- `opendata` is not part of the current artifact contract.
