use std::fs::File;
use std::io::{Cursor, Write};
use std::process::Command;
use std::sync::{Mutex, MutexGuard, OnceLock};

use flate2::{Compression, write::GzEncoder};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzWriter;
use safetensors::{Dtype, SafeTensors, tensor::TensorView, tensor::serialize_to_file};
use serde_json::Value;
use tempfile::tempdir;

fn cli_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn write_npz_gz(
    path: &std::path::Path,
    arrays: Vec<(&str, Array2<f32>)>,
    indices: Option<Vec<(&str, Array1<i64>)>>,
) {
    let mut writer = NpzWriter::new(Cursor::new(Vec::<u8>::new()));
    for (name, array) in arrays {
        writer.add_array(name, &array).expect("write array");
    }
    if let Some(indices) = indices {
        for (name, array) in indices {
            writer.add_array(name, &array).expect("write index array");
        }
    }
    let cursor = writer.finish().expect("finish artifact");
    let file = File::create(path).expect("create artifact");
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder
        .write_all(&cursor.into_inner())
        .expect("compress artifact");
    encoder.finish().expect("finish gzip");
}

fn build_fixture_dir() -> tempfile::TempDir {
    let temp_dir = tempdir().expect("tempdir");
    let data_dir = temp_dir.path();

    write_npz_gz(
        &data_dir.join("senders_receivers_encoder.npz.gz"),
        Vec::new(),
        Some(vec![
            ("senders", Array1::from_vec(vec![0_i64, 1])),
            ("receivers", Array1::from_vec(vec![1_i64, 0])),
        ]),
    );
    write_npz_gz(
        &data_dir.join("senders_receivers_processor.npz.gz"),
        Vec::new(),
        Some(vec![
            ("senders", Array1::from_vec(vec![0_i64])),
            ("receivers", Array1::from_vec(vec![0_i64])),
        ]),
    );
    write_npz_gz(
        &data_dir.join("senders_receivers_decoder.npz.gz"),
        Vec::new(),
        Some(vec![
            ("senders", Array1::from_vec(vec![0_i64, 1])),
            ("receivers", Array1::from_vec(vec![1_i64, 0])),
        ]),
    );

    write_npz_gz(
        &data_dir.join(
            "node_features_n71042_e112246_s-8416688801745003395_r-6736346125390000850.npz.gz",
        ),
        vec![
            ("coslat", Array2::<f32>::zeros((71042, 1))),
            ("sinlat", Array2::<f32>::zeros((71042, 1))),
            ("coslon", Array2::<f32>::zeros((71042, 1))),
            ("sinlon", Array2::<f32>::zeros((71042, 1))),
        ],
        None,
    );
    write_npz_gz(
        &data_dir
            .join("node_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz"),
        vec![
            ("coslat", Array2::<f32>::zeros((5882, 1))),
            ("sinlat", Array2::<f32>::zeros((5882, 1))),
            ("coslon", Array2::<f32>::zeros((5882, 1))),
            ("sinlon", Array2::<f32>::zeros((5882, 1))),
        ],
        None,
    );
    write_npz_gz(
        &data_dir.join(
            "node_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz",
        ),
        vec![
            ("coslat", Array2::<f32>::zeros((71042, 1))),
            ("sinlat", Array2::<f32>::zeros((71042, 1))),
            ("coslon", Array2::<f32>::zeros((71042, 1))),
            ("sinlon", Array2::<f32>::zeros((71042, 1))),
        ],
        None,
    );
    write_npz_gz(
        &data_dir.join(
            "edge_features_n71042_e112246_s-8416688801745003395_r-6736346125390000850.npz.gz",
        ),
        vec![("local_coords_row", Array2::<f32>::zeros((2, 3)))],
        None,
    );
    write_npz_gz(
        &data_dir
            .join("edge_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz"),
        vec![("local_coords_row", Array2::<f32>::zeros((1, 3)))],
        None,
    );
    write_npz_gz(
        &data_dir.join(
            "edge_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz",
        ),
        vec![("local_coords_row", Array2::<f32>::zeros((2, 3)))],
        None,
    );

    write_npz_gz(
        &data_dir.join(
            "temporal_normalizer_rk-era5-data_zarr-era5_1979begin_2020end_03hr_6phys_181lat_360lon_13levels_blosc1comp_Corder_monolith.npz.gz",
        ),
        vec![
            ("means", Array2::<f32>::zeros((1, 2))),
            ("stds", Array2::<f32>::ones((1, 2))),
        ],
        None,
    );
    write_npz_gz(
        &data_dir.join("orography_landsea.npz.gz"),
        vec![
            ("orography", Array2::<f32>::zeros((181, 360))),
            ("landsea", Array2::<f32>::ones((181, 360))),
        ],
        None,
    );
    write_era5_input_nc(&data_dir.join("era5_input.nc"));

    temp_dir
}

fn write_era5_input_nc(path: &std::path::Path) {
    let mut file = netcdf::create(path).expect("create input nc");
    let _time_dim = file.add_dimension("time", 1).expect("time dim");
    let _level_dim = file.add_dimension("level", 13).expect("level dim");
    let _lat_dim = file.add_dimension("latitude", 181).expect("lat dim");
    let _lon_dim = file.add_dimension("longitude", 360).expect("lon dim");
    let values = vec![0.0_f32; 13 * 181 * 360];
    for variable_name in [
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "geopotential",
    ] {
        let mut variable = file
            .add_variable::<f32>(variable_name, &["time", "level", "latitude", "longitude"])
            .expect("variable");
        variable.put_values(&values, ..).expect("put values");
    }
}

fn push_tensor(
    tensors: &mut Vec<(String, TensorView<'static>)>,
    name: &str,
    shape: Vec<usize>,
    values: &[f32],
) {
    let bytes = Box::leak(encode_f32s(values).into_boxed_slice());
    let view = TensorView::new(Dtype::F32, shape, bytes).expect("tensor view");
    tensors.push((name.to_owned(), view));
}

fn write_safetensors_fixture(path: &std::path::Path) {
    let mut tensors = Vec::new();
    push_tensor(
        &mut tensors,
        "encoder_edge_mlp.layers.0.w",
        vec![2, 23],
        &[0.0_f32; 46],
    );
    push_tensor(
        &mut tensors,
        "encoder_edge_mlp.layers.1.w",
        vec![2, 2],
        &[0.0_f32; 4],
    );
    push_tensor(
        &mut tensors,
        "encoder_edge_mlp.layers.2.w",
        vec![2, 2],
        &[0.0_f32; 4],
    );
    push_tensor(
        &mut tensors,
        "encoder_edge_mlp.layer_norm.scale",
        vec![2],
        &[1.0_f32; 2],
    );
    push_tensor(
        &mut tensors,
        "encoder_edge_mlp.layer_norm.offset",
        vec![2],
        &[0.0_f32; 2],
    );
    push_tensor(&mut tensors, "unused.tensor", vec![1], &[42.0_f32]);
    serialize_to_file(tensors, None, path).expect("write safetensors");
    let bytes = std::fs::read(path).expect("read safetensors");
    SafeTensors::deserialize(&bytes).expect("validate safetensors");
}

fn write_complete_safetensors_fixture(path: &std::path::Path) {
    let mut tensors = Vec::new();
    for (name, input_dim, output_dim, hidden_layers, with_ln) in [
        ("encoder_edge_mlp", 23, 2, 2, true),
        ("encoder_node_mlp", 2, 2, 2, true),
        ("processor_edge_init_mlp", 3, 2, 0, true),
        ("decoder_edge_mlp", 7, 2, 2, true),
        ("decoder_node_mlp", 4, 2, 2, false),
    ] {
        let total_layers = hidden_layers + 1;
        for layer_index in 0..total_layers {
            let layer_input = if layer_index == 0 { input_dim } else { 2 };
            let layer_output = output_dim;
            push_tensor(
                &mut tensors,
                &format!("{name}.layers.{layer_index}.w"),
                vec![layer_output, layer_input],
                &vec![0.0_f32; layer_output * layer_input],
            );
            push_tensor(
                &mut tensors,
                &format!("{name}.layers.{layer_index}.b"),
                vec![layer_output],
                &vec![0.0_f32; layer_output],
            );
        }
        if with_ln {
            push_tensor(
                &mut tensors,
                &format!("{name}.layer_norm.scale"),
                vec![output_dim],
                &vec![1.0_f32; output_dim],
            );
            push_tensor(
                &mut tensors,
                &format!("{name}.layer_norm.offset"),
                vec![output_dim],
                &vec![0.0_f32; output_dim],
            );
        }
    }
    for block_index in 0..9 {
        for (name, input_dim) in [
            (format!("processor_edge_mlps.{block_index}"), 6),
            (format!("processor_node_mlps.{block_index}"), 10),
        ] {
            for layer_index in 0..3 {
                let layer_input = if layer_index == 0 { input_dim } else { 2 };
                push_tensor(
                    &mut tensors,
                    &format!("{name}.layers.{layer_index}.w"),
                    vec![2, layer_input],
                    &vec![0.0_f32; 2 * layer_input],
                );
                push_tensor(
                    &mut tensors,
                    &format!("{name}.layers.{layer_index}.b"),
                    vec![2],
                    &[0.0_f32; 2],
                );
            }
            push_tensor(
                &mut tensors,
                &format!("{name}.layer_norm.scale"),
                vec![2],
                &[1.0_f32; 2],
            );
            push_tensor(
                &mut tensors,
                &format!("{name}.layer_norm.offset"),
                vec![2],
                &[0.0_f32; 2],
            );
        }
    }
    push_tensor(
        &mut tensors,
        "input_projection.w",
        vec![2, 2],
        &[1.0_f32, 0.0, 0.0, 1.0],
    );
    push_tensor(
        &mut tensors,
        "output_projection.w",
        vec![2, 2],
        &[1.0_f32, 0.0, 0.0, 1.0],
    );
    serialize_to_file(tensors, None, path).expect("write complete safetensors");
}

fn encode_f32s(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>()
}

#[test]
fn inspect_artifacts_lists_fixture_shapes() {
    let _guard = cli_test_guard();
    let fixture_dir = build_fixture_dir();
    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "inspect-artifacts",
            "--data-dir",
            fixture_dir.path().to_str().expect("fixture path"),
        ])
        .output()
        .expect("run inspect-artifacts");

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout");
    assert!(stdout.contains("senders_receivers_encoder.npz.gz"));
    assert!(stdout.contains("values: f32 [2, 2]") || stdout.contains("senders: i64 [2]"));
}

#[test]
fn inspect_geometry_reports_expected_counts() {
    let _guard = cli_test_guard();
    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args(["inspect-geometry"])
        .output()
        .expect("run inspect-geometry");

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout");
    assert!(stdout.contains("ERA5 nodes: 65160"));
    assert!(stdout.contains("H3 nodes: 5882"));
    assert!(stdout.contains("Total nodes: 71042"));
}

#[test]
fn inspect_weights_reports_missing_and_unused_keys() {
    let _guard = cli_test_guard();
    let fixture_dir = build_fixture_dir();
    let weights_path = fixture_dir.path().join("weights.safetensors");
    write_safetensors_fixture(&weights_path);

    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "inspect-weights",
            "--weights",
            weights_path.to_str().expect("weights path"),
            "--input-channels",
            "2",
            "--output-channels",
            "2",
            "--hidden-dim",
            "2",
        ])
        .output()
        .expect("run inspect-weights");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout");
    assert!(stdout.contains("Required coverage:"));
    assert!(stdout.contains("Shape mismatches: 0"));
    assert!(stdout.contains("Missing required:"));
    assert!(stdout.contains("encoder_node_mlp.layers.0.weight"));
    assert!(stdout.contains("Unused keys:"));
    assert!(stdout.contains("unused.tensor"));
}

#[test]
fn inspect_weights_reports_shape_mismatches_for_incompatible_config() {
    let _guard = cli_test_guard();
    let fixture_dir = build_fixture_dir();
    let weights_path = fixture_dir.path().join("weights.safetensors");
    write_safetensors_fixture(&weights_path);

    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "inspect-weights",
            "--weights",
            weights_path.to_str().expect("weights path"),
            "--input-channels",
            "2",
            "--output-channels",
            "2",
            "--hidden-dim",
            "3",
        ])
        .output()
        .expect("run inspect-weights");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout");
    assert!(stdout.contains("Shape mismatches:"));
    assert!(stdout.contains("encoder_edge_mlp.layers.0.weight"));
    assert!(stdout.contains("expected [3, 3] got [2, 2]"));
}

#[test]
fn inspect_weights_supports_json_output() {
    let _guard = cli_test_guard();
    let fixture_dir = build_fixture_dir();
    let weights_path = fixture_dir.path().join("weights.safetensors");
    write_safetensors_fixture(&weights_path);

    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "inspect-weights",
            "--weights",
            weights_path.to_str().expect("weights path"),
            "--input-channels",
            "2",
            "--output-channels",
            "2",
            "--hidden-dim",
            "2",
            "--json",
        ])
        .output()
        .expect("run inspect-weights json");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout");
    let payload: Value = serde_json::from_str(&stdout).expect("json payload");
    assert_eq!(
        payload["missing_required"][0],
        Value::String("encoder_node_mlp.layers.0.weight".to_owned())
    );
    assert_eq!(payload["shape_mismatches"], Value::Array(Vec::new()));
    assert_eq!(
        payload["unused_keys"][0],
        Value::String("unused.tensor".to_owned())
    );
}

#[test]
fn inspect_weights_strict_mode_fails_on_missing_required() {
    let _guard = cli_test_guard();
    let fixture_dir = build_fixture_dir();
    let weights_path = fixture_dir.path().join("weights.safetensors");
    write_safetensors_fixture(&weights_path);

    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "inspect-weights",
            "--weights",
            weights_path.to_str().expect("weights path"),
            "--input-channels",
            "2",
            "--output-channels",
            "2",
            "--hidden-dim",
            "2",
            "--strict",
        ])
        .output()
        .expect("run inspect-weights strict");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("stderr");
    assert!(stderr.contains("strict weight inspection failed"));
    assert!(stderr.contains("missing required"));
}

#[test]
fn inspect_weights_strict_mode_with_json_still_prints_report() {
    let _guard = cli_test_guard();
    let fixture_dir = build_fixture_dir();
    let weights_path = fixture_dir.path().join("weights.safetensors");
    write_safetensors_fixture(&weights_path);

    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "inspect-weights",
            "--weights",
            weights_path.to_str().expect("weights path"),
            "--input-channels",
            "2",
            "--output-channels",
            "2",
            "--hidden-dim",
            "2",
            "--json",
            "--strict",
        ])
        .output()
        .expect("run inspect-weights strict json");

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout");
    let payload: Value = serde_json::from_str(&stdout).expect("json payload");
    assert!(payload["missing_required"].is_array());
}

#[test]
fn forecast_requires_weights() {
    let _guard = cli_test_guard();
    let fixture_dir = build_fixture_dir();
    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "forecast",
            "--data-dir",
            fixture_dir.path().to_str().expect("fixture path"),
            "--init",
            "2020-01-01T00:00:00Z",
            "--steps",
            "1",
            "--input",
            "era5",
            "--out",
            fixture_dir
                .path()
                .join("forecast.nc")
                .to_str()
                .expect("output path"),
        ])
        .output()
        .expect("run forecast");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("stderr");
    assert!(stderr.contains("forecast requires --weights"));
}

#[test]
fn forecast_writes_netcdf_when_weights_are_valid() {
    let _guard = cli_test_guard();
    let fixture_dir = build_fixture_dir();
    let weights_path = fixture_dir.path().join("weights.safetensors");
    write_complete_safetensors_fixture(&weights_path);
    let output_path = fixture_dir.path().join("forecast.nc");

    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "forecast",
            "--data-dir",
            fixture_dir.path().to_str().expect("fixture path"),
            "--weights",
            weights_path.to_str().expect("weights path"),
            "--init",
            "2020-01-01T00:00:00Z",
            "--steps",
            "1",
            "--input",
            "era5",
            "--out",
            output_path.to_str().expect("output path"),
            "--input-channels",
            "2",
            "--output-channels",
            "2",
            "--hidden-dim",
            "2",
        ])
        .output()
        .expect("run forecast");

    assert!(output.status.success());
    assert!(output_path.exists());

    let file = netcdf::open(&output_path).expect("open forecast nc");
    assert_eq!(
        file.dimension("time").expect("time dim").len(),
        2,
        "expected init state plus one forecast step"
    );
    assert_eq!(file.dimension("level").expect("level dim").len(), 13);
    assert_eq!(file.dimension("latitude").expect("latitude dim").len(), 181);
    assert_eq!(
        file.dimension("longitude").expect("longitude dim").len(),
        360
    );

    let time = file.variable("time").expect("time variable");
    assert_eq!(
        time.attribute_value("units")
            .transpose()
            .expect("read time units")
            .map(String::try_from)
            .transpose()
            .expect("time units string conversion")
            .expect("time units attr"),
        "hours since 2020-01-01 00:00:00"
    );
    assert_eq!(
        time.attribute_value("calendar")
            .transpose()
            .expect("read calendar attr")
            .map(String::try_from)
            .transpose()
            .expect("calendar string conversion")
            .expect("calendar attr"),
        "standard"
    );

    for variable_name in [
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "geopotential",
    ] {
        let variable = file.variable(variable_name).expect("forecast variable");
        let dims = variable.dimensions();
        let shape = dims.iter().map(netcdf::Dimension::len).collect::<Vec<_>>();
        assert_eq!(shape, vec![2, 13, 181, 360], "{variable_name} dims");
    }
}
