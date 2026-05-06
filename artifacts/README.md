# Artifact Contract

`weathergraph-rs` expects large model and graph artifacts to live outside the repository. The Rust code consumes:

- graph connectivity `.npz.gz` files
- node and edge feature `.npz.gz` files
- temporal normalizer `.npz.gz`
- `orography_landsea.npz.gz`
- exported model weights in `safetensors`

## Upstream Source

The reference data layout comes from:

- <https://github.com/rkeisler/keisler-2022>

The upstream Python repository currently stores the artifacts in `src/keisler_2022/data/`.

## Weight Export Contract

Use `tools/export_weights.py` against the original upstream `.pkl` weight file. The exporter should:

1. Load the Haiku pickle from the upstream project.
2. Flatten the parameter tree.
3. Emit a single `safetensors` file.
4. Preserve `f32` dtype.
5. Preserve original tensor shapes.

Tensor naming convention expected by the current Rust loader:

- `encoder_edge_mlp.layers.0.weight`
- `encoder_edge_mlp.layers.0.bias`
- `encoder_edge_mlp.layers.1.weight`
- `encoder_edge_mlp.layers.1.bias`
- `encoder_edge_mlp.layer_norm.weight`
- `encoder_edge_mlp.layer_norm.bias`

Repeat the same structure for:

- `encoder_node_mlp`
- `processor_edge_mlp`
- `processor_node_mlp`
- `decoder_edge_mlp`
- `decoder_node_mlp`

The current scaffold assumes square hidden layers of `hidden_dim x hidden_dim` and bias vectors of length `hidden_dim`. Exact upstream parity work will extend this contract if needed.

## Recommended Local Layout

```text
data/
├── edge_features_*.npz.gz
├── node_features_*.npz.gz
├── orography_landsea.npz.gz
├── senders_receivers_decoder.npz.gz
├── senders_receivers_encoder.npz.gz
├── senders_receivers_processor.npz.gz
├── temporal_normalizer*.npz.gz
└── weights.safetensors
```

## Notes

- The Rust side does not read Python pickle directly.
- Large upstream artifacts should not be committed into this repository.
- Tiny synthetic test fixtures are acceptable for tests.

