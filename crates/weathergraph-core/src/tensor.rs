use candle_core::Tensor;

use crate::error::{Result, WeatherGraphError};

pub fn shape_string(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub fn gather_rows(tensor: &Tensor, indices: &[u32]) -> Result<Tensor> {
    let (n_rows, _) = tensor.dims2()?;
    for &index in indices {
        if usize::try_from(index).unwrap_or(usize::MAX) >= n_rows {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "gather_rows".to_owned(),
                expected: format!("index < {n_rows}"),
                actual: index.to_string(),
            });
        }
    }
    let indexes = Tensor::from_vec(indices.to_vec(), indices.len(), tensor.device())?;
    tensor.index_select(&indexes, 0).map_err(Into::into)
}

pub fn aggregate_receivers(
    edge_features: &Tensor,
    receivers: &[u32],
    n_nodes: usize,
) -> Result<Tensor> {
    let (n_edges, feature_dim) = edge_features.dims2()?;
    if n_edges != receivers.len() {
        return Err(WeatherGraphError::ShapeMismatch {
            name: "aggregate_receivers".to_owned(),
            expected: n_edges.to_string(),
            actual: receivers.len().to_string(),
        });
    }
    for &receiver in receivers {
        if usize::try_from(receiver).unwrap_or(usize::MAX) >= n_nodes {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "aggregate_receivers".to_owned(),
                expected: format!("receiver < {n_nodes}"),
                actual: receiver.to_string(),
            });
        }
    }
    let mut expanded = Vec::with_capacity(receivers.len() * feature_dim);
    for &receiver in receivers {
        expanded.extend(std::iter::repeat_n(receiver, feature_dim));
    }
    let indexes = Tensor::from_vec(
        expanded,
        (receivers.len(), feature_dim),
        edge_features.device(),
    )?;
    let zeros = Tensor::zeros(
        (n_nodes, feature_dim),
        edge_features.dtype(),
        edge_features.device(),
    )?;
    zeros
        .scatter_add(&indexes, edge_features, 0)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{aggregate_receivers, gather_rows};

    #[test]
    fn aggregates_receiver_features() {
        let tensor =
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).expect("tensor");
        let aggregated = aggregate_receivers(&tensor, &[0, 1], 2).expect("aggregate");
        let values = aggregated.to_vec2::<f32>().expect("to vec");
        assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn gathers_rows_by_index() {
        let tensor =
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).expect("tensor");
        let gathered = gather_rows(&tensor, &[1]).expect("gather");
        let values = gathered.to_vec2::<f32>().expect("to vec");
        assert_eq!(values, vec![vec![3.0, 4.0]]);
    }
}
