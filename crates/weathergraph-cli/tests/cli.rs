use std::fs::File;
use std::io::{Cursor, Write};
use std::process::Command;

use flate2::{Compression, write::GzEncoder};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzWriter;
use safetensors::{Dtype, SafeTensors, tensor::TensorView, tensor::serialize_to_file};
use tempfile::tempdir;

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
        vec![("local_coords", Array2::<f32>::zeros((2, 2)))],
        None,
    );
    write_npz_gz(
        &data_dir
            .join("edge_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz"),
        vec![("local_coords", Array2::<f32>::zeros((1, 2)))],
        None,
    );
    write_npz_gz(
        &data_dir.join(
            "edge_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz",
        ),
        vec![("local_coords", Array2::<f32>::zeros((2, 2)))],
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

    temp_dir
}

fn write_safetensors_fixture(path: &std::path::Path) {
    let layer0_weight = encode_f32s(&[0.0_f32, 1.0, 1.0, 0.0]);
    let layer1_weight = encode_f32s(&[1.0_f32, 0.0, 0.0, 1.0]);
    let scale = encode_f32s(&[1.0_f32, 1.0]);
    let offset = encode_f32s(&[0.0_f32, 0.0]);
    let unused = encode_f32s(&[42.0_f32]);

    let tensors = vec![
        (
            "encoder_edge_mlp.layers.0.w".to_owned(),
            TensorView::new(Dtype::F32, vec![2, 2], &layer0_weight).expect("layer0"),
        ),
        (
            "encoder_edge_mlp.layers.1.w".to_owned(),
            TensorView::new(Dtype::F32, vec![2, 2], &layer1_weight).expect("layer1"),
        ),
        (
            "encoder_edge_mlp.layer_norm.scale".to_owned(),
            TensorView::new(Dtype::F32, vec![2], &scale).expect("scale"),
        ),
        (
            "encoder_edge_mlp.layer_norm.offset".to_owned(),
            TensorView::new(Dtype::F32, vec![2], &offset).expect("offset"),
        ),
        (
            "unused.tensor".to_owned(),
            TensorView::new(Dtype::F32, vec![1], &unused).expect("unused"),
        ),
    ];
    serialize_to_file(tensors, None, path).expect("write safetensors");
    let bytes = std::fs::read(path).expect("read safetensors");
    SafeTensors::deserialize(&bytes).expect("validate safetensors");
}

fn encode_f32s(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>()
}

#[test]
fn inspect_artifacts_lists_fixture_shapes() {
    let fixture_dir = build_fixture_dir();
    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "inspect-artifacts",
            "--data-dir",
            fixture_dir.path().to_str().expect("fixture path"),
        ])
        .output()
        .expect("run inspect-artifacts");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout");
    assert!(stdout.contains("senders_receivers_encoder.npz.gz"));
    assert!(stdout.contains("values: f32 [2, 2]") || stdout.contains("senders: i64 [2]"));
}

#[test]
fn inspect_geometry_reports_expected_counts() {
    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args(["inspect-geometry"])
        .output()
        .expect("run inspect-geometry");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout");
    assert!(stdout.contains("ERA5 nodes: 65160"));
    assert!(stdout.contains("H3 nodes: 5882"));
    assert!(stdout.contains("Total nodes: 71042"));
}

#[test]
fn inspect_weights_reports_missing_and_unused_keys() {
    let fixture_dir = build_fixture_dir();
    let weights_path = fixture_dir.path().join("weights.safetensors");
    write_safetensors_fixture(&weights_path);

    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "inspect-weights",
            "--weights",
            weights_path.to_str().expect("weights path"),
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
    let fixture_dir = build_fixture_dir();
    let weights_path = fixture_dir.path().join("weights.safetensors");
    write_safetensors_fixture(&weights_path);

    let output = Command::new(env!("CARGO_BIN_EXE_weathergraph"))
        .args([
            "inspect-weights",
            "--weights",
            weights_path.to_str().expect("weights path"),
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
fn forecast_requires_weights() {
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
