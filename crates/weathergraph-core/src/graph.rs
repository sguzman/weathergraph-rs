use std::path::Path;

use candle_core::{Device, Tensor};

use crate::config::DataConfig;
use crate::error::{Result, WeatherGraphError};
use crate::features::FeatureBundle;
use crate::geometry::{ERA5_NODE_COUNT, H3_NODE_COUNT, TOTAL_NODE_COUNT};

#[derive(Debug, Clone)]
pub struct StaticNodeFeatureTensors {
    pub coslat: Tensor,
    pub sinlat: Tensor,
    pub coslon: Tensor,
    pub sinlon: Tensor,
    pub inv_n_senders: Tensor,
    pub inv_n_receivers: Tensor,
}

#[derive(Debug, Clone)]
pub struct StaticEdgeFeatureTensors {
    pub local_coords_row: Tensor,
}

#[derive(Debug, Clone)]
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
    pub node_tensors: StaticNodeFeatureTensors,
    pub edge_tensors: StaticEdgeFeatureTensors,
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
    let node_tensors = load_node_tensors(&node_features, n_nodes, &senders, &receivers)?;
    let edge_tensors = load_edge_tensors(&edge_features, senders.len())?;

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
        node_tensors,
        edge_tensors,
    })
}

fn load_node_tensors(
    bundle: &FeatureBundle,
    n_nodes: usize,
    senders: &[u32],
    receivers: &[u32],
) -> Result<StaticNodeFeatureTensors> {
    let coslat = load_or_zeros(bundle, &["coslat"], n_nodes)?;
    let sinlat = load_or_zeros(bundle, &["sinlat"], n_nodes)?;
    let coslon = load_or_zeros(bundle, &["coslon"], n_nodes)?;
    let sinlon = load_or_zeros(bundle, &["sinlon"], n_nodes)?;
    let inv_n_senders = inverse_counts(receivers, n_nodes)?;
    let inv_n_receivers = inverse_counts(senders, n_nodes)?;

    Ok(StaticNodeFeatureTensors {
        coslat,
        sinlat,
        coslon,
        sinlon,
        inv_n_senders,
        inv_n_receivers,
    })
}

fn load_edge_tensors(bundle: &FeatureBundle, n_edges: usize) -> Result<StaticEdgeFeatureTensors> {
    let local_coords_row = load_or_zeros(bundle, &["local_coords_row", "local_coords"], n_edges)?;
    Ok(StaticEdgeFeatureTensors { local_coords_row })
}

fn load_or_zeros(
    bundle: &FeatureBundle,
    candidates: &[&str],
    expected_rows: usize,
) -> Result<Tensor> {
    let array = match bundle.get_f32(candidates) {
        Ok(array) => array,
        Err(_) => ndarray::Array2::<f32>::zeros((expected_rows, 1)).into_dyn(),
    };

    let shape = array.shape().to_vec();
    if shape.is_empty() {
        return Err(WeatherGraphError::ShapeMismatch {
            name: candidates.join("/"),
            expected: "at least 1 dimension".to_owned(),
            actual: "scalar".to_owned(),
        });
    }

    if shape[0] != expected_rows {
        return Err(WeatherGraphError::ShapeMismatch {
            name: candidates.join("/"),
            expected: expected_rows.to_string(),
            actual: shape[0].to_string(),
        });
    }

    let width = shape.iter().skip(1).product::<usize>().max(1);
    let values = array.iter().copied().collect::<Vec<_>>();
    Ok(Tensor::from_vec(
        values,
        (expected_rows, width),
        &Device::Cpu,
    )?)
}

fn inverse_counts(indices: &[u32], n_nodes: usize) -> Result<Tensor> {
    let mut counts = vec![0.0_f32; n_nodes];
    for &index in indices {
        let index = usize::try_from(index)
            .map_err(|_| WeatherGraphError::InvalidConfig("index does not fit usize".to_owned()))?;
        if index >= n_nodes {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "inverse_counts".to_owned(),
                expected: format!("index < {n_nodes}"),
                actual: index.to_string(),
            });
        }
        counts[index] += 1.0;
    }
    for count in &mut counts {
        *count = if *count > 0.0 { 1.0 / *count } else { 0.0 };
    }
    Ok(Tensor::from_vec(counts, (n_nodes, 1), &Device::Cpu)?)
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
