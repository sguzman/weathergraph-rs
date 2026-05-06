use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Result, WeatherGraphError};
use crate::features::{FeatureBundle, find_artifact_path};
use crate::geometry::{ERA5_NODE_COUNT, H3_NODE_COUNT, TOTAL_NODE_COUNT};

#[derive(Debug, Clone, PartialEq)]
pub struct StaticGraph {
    pub name: String,
    pub n_nodes: usize,
    pub n_edges: usize,
    pub senders: Vec<u32>,
    pub receivers: Vec<u32>,
    pub node_features: FeatureBundle,
    pub edge_features: FeatureBundle,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphSet {
    pub encoder: StaticGraph,
    pub processor: StaticGraph,
    pub decoder: StaticGraph,
    pub n_era5_nodes: usize,
    pub n_h3_nodes: usize,
    pub n_total_nodes: usize,
}

impl GraphSet {
    pub fn load(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref();
        let encoder = load_graph(data_dir, GraphKind::Encoder, TOTAL_NODE_COUNT)?;
        let processor = load_graph(data_dir, GraphKind::Processor, H3_NODE_COUNT)?;
        let decoder = load_graph(data_dir, GraphKind::Decoder, TOTAL_NODE_COUNT)?;

        Ok(Self {
            encoder,
            processor,
            decoder,
            n_era5_nodes: ERA5_NODE_COUNT,
            n_h3_nodes: H3_NODE_COUNT,
            n_total_nodes: TOTAL_NODE_COUNT,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphKind {
    Encoder,
    Processor,
    Decoder,
}

impl GraphKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::Processor => "processor",
            Self::Decoder => "decoder",
        }
    }
}

fn load_graph(data_dir: &Path, kind: GraphKind, n_nodes: usize) -> Result<StaticGraph> {
    let senders_receivers_path = find_artifact_path(
        data_dir,
        &[&format!("senders_receivers_{}.npz.gz", kind.as_str())],
    )?;
    let senders_receivers = FeatureBundle::load(&senders_receivers_path)?;

    let senders = select_named_array(&senders_receivers, &["senders"])?
        .data
        .to_u32_vec("senders")?;
    let receivers = select_named_array(&senders_receivers, &["receivers"])?
        .data
        .to_u32_vec("receivers")?;

    if senders.len() != receivers.len() {
        return Err(WeatherGraphError::ShapeMismatch {
            name: format!("{} senders/receivers", kind.as_str()),
            expected: senders.len().to_string(),
            actual: receivers.len().to_string(),
        });
    }

    let node_features =
        FeatureBundle::load(resolve_feature_file(data_dir, kind, "node_features")?)?;
    let edge_features =
        FeatureBundle::load(resolve_feature_file(data_dir, kind, "edge_features")?)?;

    Ok(StaticGraph {
        name: kind.as_str().to_owned(),
        n_nodes,
        n_edges: senders.len(),
        senders,
        receivers,
        node_features,
        edge_features,
    })
}

fn resolve_feature_file(data_dir: &Path, kind: GraphKind, prefix: &str) -> Result<PathBuf> {
    let preferred = data_dir.join(format!("{prefix}_{}.npz.gz", kind.as_str()));
    if preferred.exists() {
        return Ok(preferred);
    }

    let mut matches = fs::read_dir(data_dir)?
        .map(|entry| entry.map(|value| value.path()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    matches.retain(|path| {
        path.file_name().is_some_and(|name| {
            let name = name.to_string_lossy();
            name.starts_with(prefix) && name.ends_with(".npz.gz")
        })
    });
    matches.sort();

    match kind {
        GraphKind::Processor => matches
            .iter()
            .find(|path| path.to_string_lossy().contains("processor"))
            .or_else(|| {
                matches
                    .iter()
                    .find(|path| path.to_string_lossy().contains("n5882"))
            })
            .cloned(),
        GraphKind::Encoder => {
            let mut total_matches = matches
                .iter()
                .filter(|path| !path.to_string_lossy().contains("processor"))
                .cloned()
                .collect::<Vec<_>>();
            total_matches.sort();
            total_matches.first().cloned()
        }
        GraphKind::Decoder => {
            let mut total_matches = matches
                .iter()
                .filter(|path| !path.to_string_lossy().contains("processor"))
                .cloned()
                .collect::<Vec<_>>();
            total_matches.sort();
            total_matches
                .get(1)
                .cloned()
                .or_else(|| total_matches.first().cloned())
        }
    }
    .ok_or_else(|| WeatherGraphError::MissingArtifact {
        name: format!("{prefix}_{}", kind.as_str()),
        path: data_dir.to_path_buf(),
    })
}

fn select_named_array<'a>(
    bundle: &'a FeatureBundle,
    candidates: &[&str],
) -> Result<&'a crate::features::NamedArray> {
    for candidate in candidates {
        if let Some(entry) = bundle
            .arrays
            .iter()
            .find(|entry| entry.stem().eq_ignore_ascii_case(candidate))
        {
            return Ok(entry);
        }
    }

    bundle
        .arrays
        .first()
        .ok_or_else(|| WeatherGraphError::MissingArtifact {
            name: format!("array {}", candidates.join("/")),
            path: bundle.source.clone(),
        })
}
