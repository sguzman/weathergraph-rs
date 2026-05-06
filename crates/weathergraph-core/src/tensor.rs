use candle_core::{Device, Tensor};

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
    let values = tensor.to_vec2::<f32>()?;
    let feature_dim = values.first().map_or(0, Vec::len);
    let mut gathered = Vec::with_capacity(indices.len() * feature_dim);

    for &index in indices {
        let row = values
            .get(index as usize)
            .ok_or_else(|| WeatherGraphError::ShapeMismatch {
                name: "gather_rows".to_owned(),
                expected: format!("index < {}", values.len()),
                actual: index.to_string(),
            })?;
        gathered.extend_from_slice(row);
    }

    Ok(Tensor::from_vec(
        gathered,
        (indices.len(), feature_dim),
        &Device::Cpu,
    )?)
}

pub fn aggregate_receivers(
    edge_features: &Tensor,
    receivers: &[u32],
    n_nodes: usize,
) -> Result<Tensor> {
    let edge_values = edge_features.to_vec2::<f32>()?;
    if edge_values.len() != receivers.len() {
        return Err(WeatherGraphError::ShapeMismatch {
            name: "aggregate_receivers".to_owned(),
            expected: edge_values.len().to_string(),
            actual: receivers.len().to_string(),
        });
    }

    let feature_dim = edge_values.first().map_or(0, Vec::len);
    let mut aggregated = vec![0.0_f32; n_nodes * feature_dim];

    for (edge_index, receiver) in receivers.iter().copied().enumerate() {
        let receiver = receiver as usize;
        if receiver >= n_nodes {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "aggregate_receivers".to_owned(),
                expected: format!("receiver < {n_nodes}"),
                actual: receiver.to_string(),
            });
        }

        for feature_index in 0..feature_dim {
            aggregated[(receiver * feature_dim) + feature_index] +=
                edge_values[edge_index][feature_index];
        }
    }

    Ok(Tensor::from_vec(
        aggregated,
        (n_nodes, feature_dim),
        &Device::Cpu,
    )?)
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
