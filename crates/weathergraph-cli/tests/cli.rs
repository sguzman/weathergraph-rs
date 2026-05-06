use std::fs::File;
use std::io::{Cursor, Write};
use std::process::Command;

use flate2::{Compression, write::GzEncoder};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzWriter;
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

    for name in [
        "node_features_encoder.npz.gz",
        "node_features_processor.npz.gz",
        "node_features_decoder.npz.gz",
        "edge_features_encoder.npz.gz",
        "edge_features_processor.npz.gz",
        "edge_features_decoder.npz.gz",
        "temporal_normalizer.npz.gz",
        "orography_landsea.npz.gz",
    ] {
        write_npz_gz(
            &data_dir.join(name),
            vec![("values", Array2::<f32>::zeros((2, 2)))],
            None,
        );
    }

    temp_dir
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
