ChatGPT




https://github.com/rkeisler/keisler-2022

i want to make a rust version of this

I’ll inspect the repo’s structure and intent first, then map it to a Rust crate design: modules, dependencies, data formats, and a first implementation path.

Thought for 17s
Yes. This is a very good Rust port project, but the right first target is Rust inference, not Rust training.

The original repo is a JAX/Haiku/Jraph implementation of Keisler 2022: encoder maps ERA5 latitude/longitude fields onto an H3 hex mesh, processor runs 9 rounds of message passing, decoder maps back to lat/lon, and autoregressive rollout gives multi-day forecasts. 

The hard part is not “Rustifying Python.” The hard part is reproducing:

JAX tensor ops

Haiku parameter layout

Jraph graph message passing

Python pickle model weights

ERA5 / ECMWF data loading

NetCDF / Zarr weather output

The best version of this project would be:

keisler-rs: a Rust-native inference engine for the Keisler 2022 graph neural weather model, with a small Python compatibility/conversion tool for the original Haiku weights.

Recommended stack
Use Candle, not Burn, for the first version.

Candle is a Rust ML framework from Hugging Face focused on performance, GPU support, and ease of deployment. It supports CPU, CUDA, WASM, and file formats including safetensors, npz, ggml, and PyTorch files. 
 Its own README also explicitly frames Candle as a way to remove Python from production inference workloads, while still keeping a lightweight binary. 

For this project, that matters because you probably do not want to train the model in Rust at first. You want to load the trained model and run forecasts.

Project shape
I would structure it like this:

keisler-rs/
  Cargo.toml
  README.md
  crates/
    keisler-core/
      src/
        lib.rs
        config.rs
        tensor.rs
        graph.rs
        geometry.rs
        features.rs
        solar.rs
        normalizer.rs
        model.rs
        runner.rs
        error.rs
    keisler-cli/
      src/
        main.rs
  tools/
    export_weights.py
    inspect_original.py
  artifacts/
    README.md
  tests/
    parity_small.rs
Crate responsibilities
geometry.rs
Port these Python pieces first:

Geometry
xyz_from_latlonr
geometry_era5
indices_h3
geometry_h3
The original Python graph code defines ERA5 and H3 geometries, where each point has both spherical coordinates and 3D Cartesian coordinates. 

Rust version:

#[derive(Debug, Clone)]
pub struct Geometry {
    pub xyz: Vec<[f32; 3]>,
    pub latlonr: Vec<[f32; 3]>,
}

impl Geometry {
    pub fn len(&self) -> usize {
        self.xyz.len()
    }

    pub fn concat(&self, other: &Geometry) -> Geometry {
        let mut xyz = self.xyz.clone();
        xyz.extend_from_slice(&other.xyz);

        let mut latlonr = self.latlonr.clone();
        latlonr.extend_from_slice(&other.latlonr);

        Geometry { xyz, latlonr }
    }
}
graph.rs
Port the StaticGraph and GraphBuilder ideas.

The original repo uses three static graphs:

e = encoder graph
p = processor graph
d = decoder graph
Those graphs load precomputed sender/receiver arrays and node/edge features from .npz.gz artifacts. 

Rust type:

#[derive(Debug, Clone)]
pub struct StaticGraph {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub senders: Vec<u32>,
    pub receivers: Vec<u32>,
    pub node_features: NodeFeatures,
    pub edge_features: EdgeFeatures,
}

#[derive(Debug, Clone)]
pub struct GraphSet {
    pub encoder: StaticGraph,
    pub processor: StaticGraph,
    pub decoder: StaticGraph,
    pub n_era5_nodes: usize,
    pub n_h3_nodes: usize,
    pub n_total_nodes: usize,
}
model.rs
This is the core. Port the GNN update functions:

edge_update_fn_encoder
node_update_fn_encoder
edge_update_fn_processor
node_update_fn_processor
edge_update_fn_decoder
node_update_fn_decoder
one_step_fn
The Python model uses MLPs, ReLU, optional LayerNorm, residual updates in the processor, and then adds the decoder output as the predicted weather-field change. 

Rust version should define explicit Candle modules:

pub struct Mlp {
    layers: Vec<candle_nn::Linear>,
    layer_norm: Option<candle_nn::LayerNorm>,
}

pub struct KeislerGnn {
    encoder_edge_mlp: Mlp,
    encoder_node_mlp: Mlp,
    processor_edge_mlp: Mlp,
    processor_node_mlp: Mlp,
    decoder_edge_mlp: Mlp,
    decoder_node_mlp: Mlp,
    n_processor_blocks: usize,
}
Do not try to mimic Haiku dynamically. Make the model explicit and boring.

runner.rs
This corresponds to the Python Runner.

The original Runner validates the weather dataset, expects 6 variables, 13 pressure levels, and a (1, 13, 181, 360) shape per variable. It normalizes data, builds the graphs, computes solar/day-of-year features, loads model weights, then rolls the model forward autoregressively. 

Rust runner:

pub struct Runner {
    pub config: Config,
    pub graphs: GraphSet,
    pub normalizer: Normalizer,
    pub model: KeislerGnn,
}

impl Runner {
    pub fn run(&mut self, input: InitialConditions, steps: usize) -> Result<Forecast> {
        let mut state = self.prepare(input, steps)?;

        for step in 0..steps {
            state = self.step(state, step)?;
        }

        Ok(state.into_forecast())
    }
}
Cargo.toml starting point
[workspace]
members = [
    "crates/keisler-core",
    "crates/keisler-cli",
]
resolver = "2"

[workspace.package]
edition = "2021"
license = "CC0-1.0"
repository = "https://github.com/sguzman/keisler-rs"
For crates/keisler-core/Cargo.toml:

[package]
name = "keisler-core"
version = "0.1.0"
edition.workspace = true
license.workspace = true

[dependencies]
anyhow = "1"
thiserror = "2"
tracing = "0.1"
serde = { version = "1", features = ["derive"] }
toml = "0.8"

ndarray = "0.16"
ndarray-npy = "0.9"
flate2 = "1"
chrono = { version = "0.4", features = ["serde"] }

candle-core = "0.9"
candle-nn = "0.9"

h3o = "0.7"
For crates/keisler-cli/Cargo.toml:

[package]
name = "keisler-cli"
version = "0.1.0"
edition.workspace = true
license.workspace = true

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"
keisler-core = { path = "../keisler-core" }
The big compatibility issue: weights
The original repo loads Haiku parameters from a Python pickle file. 
 Rust should not try to read that pickle directly as the main path.

Make a converter:

tools/export_weights.py
It should load the Haiku pickle, flatten the parameter tree, and export to either:

artifacts/keisler-2022.safetensors
or:

artifacts/keisler-2022.npz
Candle supports loading safetensors and npz, so this is a sane bridge. 

The converter is not a weakness. It is exactly the right boundary: Python is used once to extract the trained model, Rust owns inference after that.

Milestone plan
Milestone 1: artifact inspector
Goal: load the shipped .npz.gz files and print shapes.

keisler inspect-artifacts --data-dir ./data
This proves Rust can read:

senders_receivers_encoder.npz.gz
senders_receivers_processor.npz.gz
senders_receivers_decoder.npz.gz
node_features_*.npz.gz
edge_features_*.npz.gz
normalizer_*.npz.gz
orography_landsea.npz.gz
Milestone 2: geometry parity
Implement ERA5 and H3 geometry.

Then compare:

n_era5_nodes = 65160
n_h3_nodes   = 5882
n_total      = 71042
The original runner asserts these exact values. 

Milestone 3: graph aggregation kernel
Implement this primitive:

fn aggregate_receivers(
    edge_features: &Tensor,
    receivers: &[u32],
    n_nodes: usize,
) -> Result<Tensor>
This replaces Jraph’s graph aggregation.

This is the heart of the project. Once you have sender gather, receiver gather, and scatter-add aggregation, the GNN becomes straightforward.

Milestone 4: MLP and LayerNorm parity
Implement:

Linear
ReLU
Linear
LayerNorm
Make a tiny Python/Rust test where both sides use the same fake weights and same fake tensor, then compare outputs.

Milestone 5: one-step forecast parity
Run one forecast step in Python and Rust from the same initial conditions. Compare tensors before writing NetCDF.

Target tolerance:

max_abs_error < 1e-4
Maybe looser at first if CUDA kernels differ.

Milestone 6: CLI
Target CLI:

keisler forecast --init 2020-01-01T00 --steps 40 --input era5 --out forecast.nc
Equivalent to the original Python command, which supports ERA5 historical initialization and ECMWF Open Data for recent forecasts. 

What makes this portfolio-worthy
This is not “I trained MNIST in Rust.” This is a serious systems/ML port:

Python/JAX research model
        |
        v
Rust-native inference engine
        |
        v
graph neural weather forecasting
        |
        v
CUDA-capable deployable binary
That says:

you understand ML model architecture,

you can port dynamic Python research code into static Rust,

you can handle tensor layouts and graph algorithms,

you can build production inference infrastructure,

you can bridge scientific data formats.

For your GitHub profile, this is much stronger than a generic neural-net toy.

README positioning
Use this as the opening:

# keisler-rs

Rust-native inference engine for the Keisler 2022 graph neural weather forecasting model.

This project ports the inference path of `rkeisler/keisler-2022` from Python/JAX/Haiku/Jraph into Rust using Candle. The goal is to preserve the original model semantics while producing a lightweight, inspectable, deployable Rust binary for running autoregressive global weather forecasts.

The initial scope is inference parity, not training.
My recommendation
Start with artifact loading + graph construction, not the neural network.

The order should be:

1. Read compressed NPZ artifacts.
2. Rebuild ERA5/H3 geometry.
3. Rebuild encoder/processor/decoder graph structs.
4. Implement graph gather/scatter aggregation.
5. Implement MLPs.
6. Convert Haiku weights to safetensors.
7. Run one-step parity.
8. Add autoregressive forecast loop.
That gives you visible progress fast and avoids getting stuck immediately on the Haiku pickle.


good name?

Best name:

weathergraph-rs

It is clear, professional, and instantly tells people what the project is: a Rust graph-neural weather model.

Other good options:

hexcast
Short, memorable, refers to H3 hex grids + forecasting. More brandable.

h3cast
Very precise, but a little too tied to H3 specifically.

meshcast
Good general name: forecasting over a mesh/graph. Probably my favorite if you want it to sound like a real tool.

baroclinic-rs
Cool meteorology/math name, but less obvious to recruiters.

synoptic-rs
Elegant weather-science name, but vague.

isobar-rs
Clean and meteorological, but does not imply graphs or ML.

aethergraph
Cooler, more mythic, but less professional.

My ranking:

meshcast

weathergraph-rs

hexcast

baroclinic-rs

synoptic-rs

For a portfolio repo, I would use:

meshcast
And the subtitle:

Rust-native graph neural weather forecasting

