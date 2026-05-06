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
        Commands::Forecast {
            data_dir,
            weights,
            init,
            steps,
            input,
            out,
        } => forecast(data_dir, weights, init, steps, input.into(), out)?,
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

fn forecast(
    data_dir: PathBuf,
    weights: Option<PathBuf>,
    init: DateTime<Utc>,
    steps: usize,
    input: ForecastInputKind,
    out: PathBuf,
) -> Result<()> {
    let mut config = Config::from_data_dir(&data_dir);
    config.artifacts = ArtifactPaths::new(data_dir);
    config.artifacts.weights_file = weights;
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
