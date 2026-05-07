use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use safetensors::SafeTensors;
use serde::Deserialize;
use weathergraph_core::config::{ArtifactPaths, Config, ForecastInputKind, ForecastRequest};
use weathergraph_core::runner::Runner;

#[derive(Debug, Deserialize)]
struct ParityFixtureManifest {
    data_dir: PathBuf,
    weights_file: PathBuf,
    init: String,
    steps: usize,
    tolerance: f32,
}

#[test]
fn one_step_fixture_matches_reference_when_present() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("parity")
        .join("one_step");

    if !fixture_dir.exists() {
        return;
    }

    let manifest_path = fixture_dir.join("manifest.json");
    if !manifest_path.exists() {
        return;
    }

    let manifest: ParityFixtureManifest =
        serde_json::from_slice(&fs::read(&manifest_path).expect("read parity manifest"))
            .expect("parse parity manifest");

    let mut config = Config::from_data_dir(&manifest.data_dir);
    config.artifacts =
        ArtifactPaths::new(manifest.data_dir.clone()).with_weights(&manifest.weights_file);
    let runner = Runner::load(config).expect("load runner");

    let request = ForecastRequest {
        init: manifest.init.parse().expect("parse init"),
        steps: manifest.steps,
        input: ForecastInputKind::Era5,
        out: fixture_dir.join("unused.nc"),
    };

    let fixture_tensors = load_fixture_tensors(&fixture_dir.join("tensors.safetensors"));
    let input = fixture_tensor(&fixture_tensors, "input_state");
    let solar = fixture_tensor(&fixture_tensors, "solar");
    let doy = fixture_tensor(&fixture_tensors, "doy");
    let orography = fixture_tensor(&fixture_tensors, "orography");
    let landsea = fixture_tensor(&fixture_tensors, "landsea");
    let expected = fixture_tensor(&fixture_tensors, "expected_output");

    // Keep the request exercised even though the parity call uses exported tensors directly.
    request.validate().expect("validate request");

    let actual = runner
        .model
        .one_step_graph(&input, &runner.graphs, &solar, &doy, &orography, &landsea)
        .expect("one-step graph");

    let max_abs_error = max_abs_error(&actual, &expected);
    assert!(
        max_abs_error <= manifest.tolerance,
        "max_abs_error={} tolerance={}",
        max_abs_error,
        manifest.tolerance
    );
}

fn load_fixture_tensors(path: &Path) -> SafeTensors<'static> {
    let bytes = fs::read(path).expect("read fixture tensors");
    let leaked = Box::leak(bytes.into_boxed_slice());
    SafeTensors::deserialize(leaked).expect("deserialize fixture tensors")
}

fn fixture_tensor(tensors: &SafeTensors<'_>, name: &str) -> Tensor {
    let view = tensors.tensor(name).expect("fixture tensor");
    let values = view
        .data()
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size")))
        .collect::<Vec<_>>();
    let shape = view.shape().to_vec();
    Tensor::from_vec(values, shape.as_slice(), &Device::Cpu).expect("tensor from fixture")
}

fn max_abs_error(lhs: &Tensor, rhs: &Tensor) -> f32 {
    let left = lhs.to_vec2::<f32>().expect("left values");
    let right = rhs.to_vec2::<f32>().expect("right values");
    left.into_iter()
        .zip(right)
        .flat_map(|(lhs_row, rhs_row)| lhs_row.into_iter().zip(rhs_row))
        .map(|(lhs_value, rhs_value)| (lhs_value - rhs_value).abs())
        .fold(0.0_f32, f32::max)
}
