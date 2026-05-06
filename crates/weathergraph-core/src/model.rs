use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use safetensors::SafeTensors;

use crate::config::ModelConfig;
use crate::error::{Result, WeatherGraphError};

#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl LinearLayer {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let inputs = input.to_vec2::<f32>()?;
        let weights = self.weight.to_vec2::<f32>()?;
        let biases = self
            .bias
            .as_ref()
            .map(Tensor::to_vec1::<f32>)
            .transpose()?
            .unwrap_or_else(|| vec![0.0; weights.len()]);
        let input_dim = inputs.first().map_or(0, Vec::len);
        let weight_input_dim = weights.first().map_or(0, Vec::len);

        if input_dim != weight_input_dim {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "linear layer".to_owned(),
                expected: format!("input width {weight_input_dim}"),
                actual: input_dim.to_string(),
            });
        }

        let mut output = Vec::with_capacity(inputs.len() * weights.len());
        for row in &inputs {
            for (out_index, weight_row) in weights.iter().enumerate() {
                let value = row
                    .iter()
                    .zip(weight_row)
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f32>()
                    + biases[out_index];
                output.push(value);
            }
        }

        Ok(Tensor::from_vec(
            output,
            (inputs.len(), weights.len()),
            &Device::Cpu,
        )?)
    }
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    #[allow(clippy::cast_precision_loss)]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let rows = input.to_vec2::<f32>()?;
        let gamma = self.weight.to_vec1::<f32>()?;
        let beta = self.bias.to_vec1::<f32>()?;
        let width = rows.first().map_or(0, Vec::len);

        if gamma.len() != width || beta.len() != width {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "layer norm".to_owned(),
                expected: format!("gamma/beta width {width}"),
                actual: format!("{}/{}", gamma.len(), beta.len()),
            });
        }

        let mut output = Vec::with_capacity(rows.len() * width);
        for row in rows {
            let mean = row.iter().sum::<f32>() / width as f32;
            let variance = row
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / width as f32;
            let denom = (variance + self.eps).sqrt();

            for (index, value) in row.iter().enumerate() {
                let normalized = (*value - mean) / denom;
                output.push((normalized * gamma[index]) + beta[index]);
            }
        }

        let batch = output.len() / width;
        Ok(Tensor::from_vec(output, (batch, width), &Device::Cpu)?)
    }
}

#[derive(Debug, Clone)]
pub struct Mlp {
    pub layers: Vec<LinearLayer>,
    pub layer_norm: Option<LayerNorm>,
}

impl Mlp {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut current = input.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            let mut next = layer.forward(&current)?;
            if index + 1 != self.layers.len() {
                next = relu(&next)?;
            }
            current = next;
        }

        if let Some(layer_norm) = &self.layer_norm {
            layer_norm.forward(&current)
        } else {
            Ok(current)
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeislerGnn {
    pub encoder_edge_mlp: Mlp,
    pub encoder_node_mlp: Mlp,
    pub processor_edge_mlp: Mlp,
    pub processor_node_mlp: Mlp,
    pub decoder_edge_mlp: Mlp,
    pub decoder_node_mlp: Mlp,
    pub n_processor_blocks: usize,
}

impl KeislerGnn {
    pub fn placeholder(config: &ModelConfig, device: &Device) -> Result<Self> {
        let hidden = config.hidden_dim;
        let mlp = Mlp {
            layers: vec![
                linear_identity(hidden, hidden, device)?,
                linear_identity(hidden, hidden, device)?,
            ],
            layer_norm: config
                .use_layer_norm
                .then(|| layer_norm_identity(hidden, device))
                .transpose()?,
        };

        Ok(Self {
            encoder_edge_mlp: mlp.clone(),
            encoder_node_mlp: mlp.clone(),
            processor_edge_mlp: mlp.clone(),
            processor_node_mlp: mlp.clone(),
            decoder_edge_mlp: mlp.clone(),
            decoder_node_mlp: mlp,
            n_processor_blocks: config.processor_blocks,
        })
    }

    pub fn from_safetensors(
        path: impl AsRef<Path>,
        config: &ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let map = tensors
            .names()
            .iter()
            .map(|name| {
                let view = tensors.tensor(name)?;
                let values = view
                    .data()
                    .chunks_exact(std::mem::size_of::<f32>())
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size")))
                    .collect::<Vec<_>>();
                Ok(((*name).to_owned(), (view.shape().to_vec(), values)))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        Self::from_weight_map(&map, config, device)
    }

    pub fn from_weight_map(
        tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        config: &ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            encoder_edge_mlp: load_mlp("encoder_edge_mlp", tensors, config, device)?,
            encoder_node_mlp: load_mlp("encoder_node_mlp", tensors, config, device)?,
            processor_edge_mlp: load_mlp("processor_edge_mlp", tensors, config, device)?,
            processor_node_mlp: load_mlp("processor_node_mlp", tensors, config, device)?,
            decoder_edge_mlp: load_mlp("decoder_edge_mlp", tensors, config, device)?,
            decoder_node_mlp: load_mlp("decoder_node_mlp", tensors, config, device)?,
            n_processor_blocks: config.processor_blocks,
        })
    }

    pub fn one_step(&self, state: &Tensor) -> Result<Tensor> {
        let mut current = self.encoder_node_mlp.forward(state)?;
        current = self.encoder_edge_mlp.forward(&current)?;

        for _ in 0..self.n_processor_blocks {
            current = self.processor_node_mlp.forward(&current)?;
            current = self.processor_edge_mlp.forward(&current)?;
        }

        current = self.decoder_edge_mlp.forward(&current)?;
        self.decoder_node_mlp.forward(&current)
    }
}

fn relu(tensor: &Tensor) -> Result<Tensor> {
    let values = tensor.to_vec2::<f32>()?;
    let width = values.first().map_or(0, Vec::len);
    let flattened = values
        .into_iter()
        .flat_map(|row| row.into_iter().map(|value| value.max(0.0)))
        .collect::<Vec<_>>();
    let batch = flattened.len() / width;
    Ok(Tensor::from_vec(flattened, (batch, width), &Device::Cpu)?)
}

fn linear_identity(input_dim: usize, output_dim: usize, device: &Device) -> Result<LinearLayer> {
    let mut weights = vec![0.0_f32; output_dim * input_dim];
    let diagonal = input_dim.min(output_dim);
    for index in 0..diagonal {
        weights[(index * input_dim) + index] = 1.0;
    }

    Ok(LinearLayer {
        weight: Tensor::from_vec(weights, (output_dim, input_dim), device)?,
        bias: Some(Tensor::zeros(output_dim, candle_core::DType::F32, device)?),
    })
}

fn layer_norm_identity(width: usize, device: &Device) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: Tensor::ones(width, candle_core::DType::F32, device)?,
        bias: Tensor::zeros(width, candle_core::DType::F32, device)?,
        eps: 1.0e-5_f32,
    })
}

fn load_mlp(
    prefix: &str,
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    config: &ModelConfig,
    device: &Device,
) -> Result<Mlp> {
    let hidden = config.hidden_dim;
    let layer_names = ["layers.0", "layers.1"];
    let mut layers = Vec::with_capacity(layer_names.len());

    for layer_name in layer_names {
        let weight_key = format!("{prefix}.{layer_name}.weight");
        let bias_key = format!("{prefix}.{layer_name}.bias");
        let weight = load_tensor(tensors, &weight_key, device)?;
        let bias = load_tensor(tensors, &bias_key, device).ok();
        layers.push(LinearLayer { weight, bias });
    }

    let layer_norm = if config.use_layer_norm {
        let weight_key = format!("{prefix}.layer_norm.weight");
        let bias_key = format!("{prefix}.layer_norm.bias");
        Some(LayerNorm {
            weight: load_tensor(tensors, &weight_key, device)?,
            bias: load_tensor(tensors, &bias_key, device)?,
            eps: 1.0e-5_f32,
        })
    } else {
        None
    };

    if layers.is_empty() {
        return Err(WeatherGraphError::InvalidConfig(format!(
            "no layers loaded for `{prefix}`"
        )));
    }

    if layer_norm.is_none() && config.use_layer_norm {
        return Err(WeatherGraphError::InvalidConfig(format!(
            "layer norm missing for `{prefix}`"
        )));
    }

    if layers[0].weight.dims2()? != (hidden, hidden) {
        return Err(WeatherGraphError::ShapeMismatch {
            name: format!("{prefix}.layers.0.weight"),
            expected: format!("[{hidden}, {hidden}]"),
            actual: format!("{:?}", layers[0].weight.dims()),
        });
    }

    Ok(Mlp { layers, layer_norm })
}

fn load_tensor(
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    key: &str,
    device: &Device,
) -> Result<Tensor> {
    let (shape, values) = tensors
        .get(key)
        .ok_or_else(|| WeatherGraphError::MissingArtifact {
            name: key.to_owned(),
            path: Path::new("weights.safetensors").to_path_buf(),
        })?;
    Ok(Tensor::from_vec(values.clone(), shape.as_slice(), device)?)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};

    use super::{KeislerGnn, Mlp, layer_norm_identity, linear_identity};
    use crate::config::ModelConfig;

    #[test]
    fn mlp_preserves_shape() {
        let device = Device::Cpu;
        let mlp = Mlp {
            layers: vec![
                linear_identity(2, 2, &device).expect("layer 0"),
                linear_identity(2, 2, &device).expect("layer 1"),
            ],
            layer_norm: Some(layer_norm_identity(2, &device).expect("layer norm")),
        };
        let input = Tensor::from_vec(vec![1.0_f32, -1.0], (1, 2), &device).expect("input");
        let output = mlp.forward(&input).expect("forward");
        assert_eq!(output.dims2().expect("dims"), (1, 2));
    }

    #[test]
    fn gnn_placeholder_runs_one_step() {
        let device = Device::Cpu;
        let config = ModelConfig {
            hidden_dim: 2,
            processor_blocks: 2,
            use_layer_norm: true,
        };
        let model = KeislerGnn::placeholder(&config, &device).expect("placeholder");
        let input = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2), &device).expect("input");
        let output = model.one_step(&input).expect("one step");
        assert_eq!(output.dims2().expect("dims"), (2, 2));
    }

    #[test]
    fn gnn_loads_fake_weight_map() {
        let device = Device::Cpu;
        let config = ModelConfig {
            hidden_dim: 2,
            processor_blocks: 1,
            use_layer_norm: true,
        };
        let mut tensors = HashMap::new();
        for prefix in [
            "encoder_edge_mlp",
            "encoder_node_mlp",
            "processor_edge_mlp",
            "processor_node_mlp",
            "decoder_edge_mlp",
            "decoder_node_mlp",
        ] {
            for layer in ["layers.0", "layers.1"] {
                tensors.insert(
                    format!("{prefix}.{layer}.weight"),
                    (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
                );
                tensors.insert(format!("{prefix}.{layer}.bias"), (vec![2], vec![0.0, 0.0]));
            }
            tensors.insert(
                format!("{prefix}.layer_norm.weight"),
                (vec![2], vec![1.0, 1.0]),
            );
            tensors.insert(
                format!("{prefix}.layer_norm.bias"),
                (vec![2], vec![0.0, 0.0]),
            );
        }

        let model = KeislerGnn::from_weight_map(&tensors, &config, &device).expect("weight map");
        let input = Tensor::from_vec(vec![1.0_f32, 2.0], (1, 2), &device).expect("input");
        let output = model.one_step(&input).expect("one step");
        assert_eq!(output.dims2().expect("dims"), (1, 2));
    }
}
