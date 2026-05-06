use std::path::Path;

use crate::config::DataConfig;
use crate::error::{Result, WeatherGraphError};
use crate::features::FeatureBundle;
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
    pub node_feature_keys: Vec<String>,
    pub edge_feature_keys: Vec<String>,
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
    pub fn load(data_dir: impl AsRef<Path>, data: &DataConfig) -> Result<Self> {
        let data_dir = data_dir.as_ref();
        let encoder = load_graph(data_dir, GraphKind::Encoder, TOTAL_NODE_COUNT, data)?;
        let processor = load_graph(data_dir, GraphKind::Processor, H3_NODE_COUNT, data)?;
        let decoder = load_graph(data_dir, GraphKind::Decoder, TOTAL_NODE_COUNT, data)?;

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

fn load_graph(
    data_dir: &Path,
    kind: GraphKind,
    n_nodes: usize,
    data: &DataConfig,
) -> Result<StaticGraph> {
    let senders_receivers_path = data_dir.join(match kind {
        GraphKind::Encoder => &data.senders_receivers_encoder,
        GraphKind::Processor => &data.senders_receivers_processor,
        GraphKind::Decoder => &data.senders_receivers_decoder,
    });
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

    let node_features = FeatureBundle::load(data_dir.join(match kind {
        GraphKind::Encoder => &data.node_features_e,
        GraphKind::Processor => &data.node_features_p,
        GraphKind::Decoder => &data.node_features_d,
    }))?;
    let edge_features = FeatureBundle::load(data_dir.join(match kind {
        GraphKind::Encoder => &data.edge_features_e,
        GraphKind::Processor => &data.edge_features_p,
        GraphKind::Decoder => &data.edge_features_d,
    }))?;
    let node_feature_keys = node_features
        .arrays
        .iter()
        .map(|entry| entry.stem().to_owned())
        .filter(|name| name != "local_coords")
        .collect::<Vec<_>>();
    let edge_feature_keys = edge_features
        .arrays
        .iter()
        .map(|entry| entry.stem().to_owned())
        .collect::<Vec<_>>();

    Ok(StaticGraph {
        name: kind.as_str().to_owned(),
        n_nodes,
        n_edges: senders.len(),
        senders,
        receivers,
        node_features,
        edge_features,
        node_feature_keys,
        edge_feature_keys,
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
