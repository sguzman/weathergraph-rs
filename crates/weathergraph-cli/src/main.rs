use std::path::PathBuf;

use anyhow::Result;
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand, ValueEnum};
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::EnvFilter;
use weathergraph_core::config::{ArtifactPaths, Config, ForecastInputKind, ForecastRequest};
use weathergraph_core::features::summarize_data_dir;
use weathergraph_core::geometry::{
    ERA5_NODE_COUNT, H3_NODE_COUNT, TOTAL_NODE_COUNT, build_era5_geometry, build_h3_geometry,
    verify_geometry_counts,
};
use weathergraph_core::graph::GraphSet;
use weathergraph_core::model::KeislerGnn;
use weathergraph_core::runner::Runner;

#[derive(Debug, Parser)]
#[command(name = "weathergraph")]
#[command(about = "Rust-native inference scaffolding for the Keisler 2022 weather GNN")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    InspectArtifacts {
        #[arg(long)]
        data_dir: PathBuf,
    },
    InspectGeometry {
        #[arg(long)]
        data_dir: Option<PathBuf>,
    },
    InspectWeights {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        strict: bool,
        #[arg(long, default_value_t = 78)]
        input_channels: usize,
        #[arg(long, default_value_t = 78)]
        output_channels: usize,
        #[arg(long, default_value_t = 256)]
        hidden_dim: usize,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        use_layer_norm: bool,
    },
    Forecast {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long)]
        weights: Option<PathBuf>,
        #[arg(long)]
        init: DateTime<Utc>,
        #[arg(long)]
        steps: usize,
        #[arg(long, value_enum, default_value_t = InputKindArg::Era5)]
        input: InputKindArg,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 78)]
        input_channels: usize,
        #[arg(long, default_value_t = 78)]
        output_channels: usize,
        #[arg(long, default_value_t = 256)]
        hidden_dim: usize,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        use_layer_norm: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum InputKindArg {
    Era5,
    Opendata,
}

impl From<InputKindArg> for ForecastInputKind {
    fn from(value: InputKindArg) -> Self {
        match value {
            InputKindArg::Era5 => Self::Era5,
            InputKindArg::Opendata => Self::Opendata,
        }
    }
}

fn main() -> Result<()> {
    init_tracing()?;
    let cli = Cli::parse();

    match cli.command {
        Commands::InspectArtifacts { data_dir } => inspect_artifacts(&data_dir)?,
        Commands::InspectGeometry { data_dir: _ } => inspect_geometry()?,
        Commands::InspectWeights {
            weights,
            json,
            strict,
            input_channels,
            output_channels,
            hidden_dim,
            use_layer_norm,
        } => inspect_weights(
            &weights,
            json,
            strict,
            input_channels,
            output_channels,
            hidden_dim,
            use_layer_norm,
        )?,
        Commands::Forecast {
            data_dir,
            weights,
            init,
            steps,
            input,
            out,
            input_channels,
            output_channels,
            hidden_dim,
            use_layer_norm,
        } => forecast(
            data_dir,
            weights,
            init,
            steps,
            input.into(),
            out,
            input_channels,
            output_channels,
            hidden_dim,
            use_layer_norm,
        )?,
    }

    Ok(())
}

fn init_tracing() -> Result<()> {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(())
}

fn inspect_artifacts(data_dir: &PathBuf) -> Result<()> {
    info!(path = %data_dir.display(), "inspecting artifacts");
    let summaries = summarize_data_dir(data_dir)?;
    for summary in summaries {
        println!("{}", summary.file_name);
        for array in summary.arrays {
            println!("  - {}: {} {:?}", array.name, array.dtype, array.shape);
        }
    }
    let graph_set = GraphSet::load(data_dir, &Config::from_data_dir(data_dir).data)?;
    println!(
        "encoder node feature keys: {:?}",
        graph_set.encoder.node_feature_keys
    );
    println!(
        "processor node feature keys: {:?}",
        graph_set.processor.node_feature_keys
    );
    println!(
        "decoder node feature keys: {:?}",
        graph_set.decoder.node_feature_keys
    );
    Ok(())
}

fn inspect_geometry() -> Result<()> {
    verify_geometry_counts()?;
    let era5 = build_era5_geometry();
    let h3 = build_h3_geometry()?;
    println!("ERA5 nodes: {ERA5_NODE_COUNT}");
    println!("H3 nodes: {H3_NODE_COUNT}");
    println!("Total nodes: {TOTAL_NODE_COUNT}");
    println!("ERA5 geometry len: {}", era5.len());
    println!("H3 geometry len: {}", h3.len());
    Ok(())
}

fn inspect_weights(
    weights: &PathBuf,
    json: bool,
    strict: bool,
    input_channels: usize,
    output_channels: usize,
    hidden_dim: usize,
    use_layer_norm: bool,
) -> Result<()> {
    info!(path = %weights.display(), "inspecting weights");
    let mut config = Config::from_data_dir(".").model;
    config.input_channels = input_channels;
    config.output_channels = output_channels;
    config.hidden_dim = hidden_dim;
    config.use_layer_norm = use_layer_norm;
    let report = KeislerGnn::inspect_safetensors(weights, &config)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return enforce_strict_weight_report(&report, strict);
    }
    println!("Weights: {}", weights.display());
    println!("Available keys: {}", report.available_keys.len());
    println!("Required coverage: {}", report.required_coverage());
    println!("Matched required: {}", report.matched_required.len());
    println!("Matched optional: {}", report.matched_optional.len());
    println!("Shape mismatches: {}", report.shape_mismatches.len());
    println!("Dtype mismatches: {}", report.dtype_mismatches.len());

    if !report.missing_required.is_empty() {
        println!("Missing required:");
        for key in &report.missing_required {
            println!("  - {key}");
        }
    }
    if !report.missing_optional.is_empty() {
        println!("Missing optional:");
        for key in &report.missing_optional {
            println!("  - {key}");
        }
    }
    if !report.shape_mismatches.is_empty() {
        println!("Shape mismatches:");
        for mismatch in &report.shape_mismatches {
            println!(
                "  - {} <- {} expected {:?} got {:?}",
                mismatch.canonical_key,
                mismatch.matched_key,
                mismatch.expected_shape,
                mismatch.actual_shape
            );
        }
    }
    if !report.dtype_mismatches.is_empty() {
        println!("Dtype mismatches:");
        for mismatch in &report.dtype_mismatches {
            println!(
                "  - {} <- {} expected {} got {}",
                mismatch.canonical_key,
                mismatch.matched_key,
                mismatch.expected_dtype,
                mismatch.actual_dtype
            );
        }
    }
    if !report.unused_keys.is_empty() {
        println!("Unused keys:");
        for key in &report.unused_keys {
            println!("  - {key}");
        }
    }
    enforce_strict_weight_report(&report, strict)
}

fn enforce_strict_weight_report(
    report: &weathergraph_core::model::WeightInspectionReport,
    strict: bool,
) -> Result<()> {
    if strict
        && (!report.missing_required.is_empty()
            || !report.shape_mismatches.is_empty()
            || !report.dtype_mismatches.is_empty())
    {
        anyhow::bail!(
            "strict weight inspection failed: {} missing required, {} shape mismatches, {} dtype mismatches",
            report.missing_required.len(),
            report.shape_mismatches.len(),
            report.dtype_mismatches.len()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forecast(
    data_dir: PathBuf,
    weights: Option<PathBuf>,
    init: DateTime<Utc>,
    steps: usize,
    input: ForecastInputKind,
    out: PathBuf,
    input_channels: usize,
    output_channels: usize,
    hidden_dim: usize,
    use_layer_norm: bool,
) -> Result<()> {
    let mut config = Config::from_data_dir(&data_dir);
    config.artifacts = ArtifactPaths::new(data_dir);
    config.artifacts.weights_file = weights;
    config.model.input_channels = input_channels;
    config.model.output_channels = output_channels;
    config.model.hidden_dim = hidden_dim;
    config.model.use_layer_norm = use_layer_norm;
    let runner = Runner::load(config)?;
    let request = ForecastRequest {
        init,
        steps,
        input,
        out,
    };
    runner.run_forecast(&request)?;
    Ok(())
}
