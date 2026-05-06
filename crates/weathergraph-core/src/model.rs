use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use safetensors::SafeTensors;

use crate::config::ModelConfig;
use crate::error::{Result, WeatherGraphError};
use crate::graph::GraphSet;
use crate::tensor::{aggregate_receivers, gather_rows};

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
    pub input_projection: LinearLayer,
    pub encoder_edge_mlp: Mlp,
    pub encoder_node_mlp: Mlp,
    pub processor_edge_init_mlp: Mlp,
    pub processor_edge_mlp: Mlp,
    pub processor_node_mlp: Mlp,
    pub decoder_edge_mlp: Mlp,
    pub decoder_node_mlp: Mlp,
    pub output_projection: LinearLayer,
    pub n_processor_blocks: usize,
}

impl KeislerGnn {
    pub fn placeholder(config: &ModelConfig, device: &Device) -> Result<Self> {
        let hidden = config.hidden_dim;
        let hidden_mlp = Mlp {
            layers: vec![
                linear_identity(hidden, hidden, device)?,
                linear_identity(hidden, hidden, device)?,
            ],
            layer_norm: config
                .use_layer_norm
                .then(|| layer_norm_identity(hidden, device))
                .transpose()?,
        };
        let decoder_node_mlp = Mlp {
            layers: vec![
                linear_identity(hidden, hidden, device)?,
                linear_identity(hidden, hidden, device)?,
            ],
            layer_norm: None,
        };

        Ok(Self {
            input_projection: linear_identity(config.input_channels, hidden, device)?,
            encoder_edge_mlp: hidden_mlp.clone(),
            encoder_node_mlp: hidden_mlp.clone(),
            processor_edge_init_mlp: hidden_mlp.clone(),
            processor_edge_mlp: hidden_mlp.clone(),
            processor_node_mlp: hidden_mlp.clone(),
            decoder_edge_mlp: hidden_mlp,
            decoder_node_mlp,
            output_projection: linear_identity(hidden, config.output_channels, device)?,
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
        let processor_edge_mlp = load_mlp_with_aliases(
            &["processor_edge_mlp"],
            tensors,
            config.hidden_dim,
            config.hidden_dim,
            config.use_layer_norm,
            device,
        )?;
        Ok(Self {
            input_projection: load_linear_or_identity(
                "input_projection",
                tensors,
                config.input_channels,
                config.hidden_dim,
                device,
            )?,
            encoder_edge_mlp: load_mlp_with_aliases(
                &["encoder_edge_mlp"],
                tensors,
                config.hidden_dim,
                config.hidden_dim,
                config.use_layer_norm,
                device,
            )?,
            encoder_node_mlp: load_mlp_with_aliases(
                &["encoder_node_mlp"],
                tensors,
                config.hidden_dim,
                config.hidden_dim,
                config.use_layer_norm,
                device,
            )?,
            processor_edge_init_mlp: load_mlp_optional(
                &[
                    "processor_edge_init_mlp",
                    "processor_edge_mlp_init",
                    "net_edges",
                ],
                tensors,
                config.hidden_dim,
                true,
                device,
            )?
            .unwrap_or_else(|| processor_edge_mlp.clone()),
            processor_edge_mlp,
            processor_node_mlp: load_mlp_with_aliases(
                &["processor_node_mlp"],
                tensors,
                config.hidden_dim,
                config.hidden_dim,
                config.use_layer_norm,
                device,
            )?,
            decoder_edge_mlp: load_mlp_with_aliases(
                &["decoder_edge_mlp"],
                tensors,
                config.hidden_dim,
                config.hidden_dim,
                config.use_layer_norm,
                device,
            )?,
            decoder_node_mlp: load_mlp_with_aliases(
                &["decoder_node_mlp"],
                tensors,
                config.hidden_dim,
                config.hidden_dim,
                false,
                device,
            )?,
            output_projection: load_linear_or_identity(
                "output_projection",
                tensors,
                config.hidden_dim,
                config.output_channels,
                device,
            )?,
            n_processor_blocks: config.processor_blocks,
        })
    }

    pub fn one_step(&self, state: &Tensor) -> Result<Tensor> {
        let mut current = self.input_projection.forward(state)?;
        current = self.encoder_node_mlp.forward(&current)?;
        current = self.encoder_edge_mlp.forward(&current)?;
        current = self.processor_edge_init_mlp.forward(&current)?;

        for _ in 0..self.n_processor_blocks {
            current = self.processor_node_mlp.forward(&current)?;
            current = self.processor_edge_mlp.forward(&current)?;
        }

        current = self.decoder_edge_mlp.forward(&current)?;
        current = self.decoder_node_mlp.forward(&current)?;
        self.output_projection.forward(&current)
    }

    pub fn one_step_graph(
        &self,
        state: &Tensor,
        graphs: &GraphSet,
        solar: &Tensor,
        doy: &Tensor,
        orography: &Tensor,
        landsea: &Tensor,
    ) -> Result<Tensor> {
        let projected_state = self.input_projection.forward(state)?;

        let sender_state = gather_rows(&projected_state, &graphs.encoder.senders)?;
        let sender_solar = gather_rows(solar, &graphs.encoder.senders)?;
        let sender_orography = gather_rows(orography, &graphs.encoder.senders)?;
        let sender_landsea = gather_rows(landsea, &graphs.encoder.senders)?;
        let sender_coslat =
            gather_rows(&graphs.encoder.node_tensors.coslat, &graphs.encoder.senders)?;
        let sender_sinlat =
            gather_rows(&graphs.encoder.node_tensors.sinlat, &graphs.encoder.senders)?;
        let sender_coslon =
            gather_rows(&graphs.encoder.node_tensors.coslon, &graphs.encoder.senders)?;
        let sender_sinlon =
            gather_rows(&graphs.encoder.node_tensors.sinlon, &graphs.encoder.senders)?;
        let sender_doy = gather_rows(doy, &graphs.encoder.senders)?;

        let encoder_edge_input = concat_tensors(&[
            &graphs.encoder.edge_tensors.local_coords_row,
            &sender_state,
            &sender_solar,
            &sender_orography,
            &sender_landsea,
            &sender_coslat,
            &sender_sinlat,
            &sender_coslon,
            &sender_sinlon,
            &sender_doy,
        ])?;
        let encoder_edge_hidden = self.encoder_edge_mlp.forward(&encoder_edge_input)?;
        let encoder_agg = aggregate_receivers(
            &encoder_edge_hidden,
            &graphs.encoder.receivers,
            graphs.encoder.n_nodes,
        )?;
        let encoder_node_input =
            elementwise_mul(&encoder_agg, &graphs.encoder.node_tensors.inv_n_senders)?;
        let encoder_hidden = self.encoder_node_mlp.forward(&encoder_node_input)?;

        let mut processor_edge_hidden = self
            .processor_edge_init_mlp
            .forward(&graphs.processor.edge_tensors.local_coords_row)?;
        let mut processor_node_hidden =
            slice_rows(&encoder_hidden, graphs.n_era5_nodes, graphs.n_h3_nodes)?;

        for _ in 0..self.n_processor_blocks {
            let sender_node_hidden =
                gather_rows(&processor_node_hidden, &graphs.processor.senders)?;
            let receiver_node_hidden =
                gather_rows(&processor_node_hidden, &graphs.processor.receivers)?;
            let processor_edge_input = concat_tensors(&[
                &processor_edge_hidden,
                &sender_node_hidden,
                &receiver_node_hidden,
            ])?;
            let processor_edge_delta = self.processor_edge_mlp.forward(&processor_edge_input)?;
            processor_edge_hidden = add_tensors(&processor_edge_hidden, &processor_edge_delta)?;

            let agg_sender = aggregate_receivers(
                &processor_edge_hidden,
                &graphs.processor.senders,
                graphs.processor.n_nodes,
            )?;
            let agg_receiver = aggregate_receivers(
                &processor_edge_hidden,
                &graphs.processor.receivers,
                graphs.processor.n_nodes,
            )?;
            let processor_node_input = concat_tensors(&[
                &processor_node_hidden,
                &graphs.processor.node_tensors.coslat,
                &graphs.processor.node_tensors.sinlat,
                &graphs.processor.node_tensors.coslon,
                &graphs.processor.node_tensors.sinlon,
                &elementwise_mul(&agg_sender, &graphs.processor.node_tensors.inv_n_receivers)?,
                &elementwise_mul(&agg_receiver, &graphs.processor.node_tensors.inv_n_senders)?,
            ])?;
            let processor_node_delta = self.processor_node_mlp.forward(&processor_node_input)?;
            processor_node_hidden = add_tensors(&processor_node_hidden, &processor_node_delta)?;
        }

        let (_, hidden_dim) = processor_node_hidden.dims2()?;
        let full_hidden = pad_h3_hidden(
            &processor_node_hidden,
            graphs.n_total_nodes,
            graphs.n_era5_nodes,
            hidden_dim,
        )?;
        let decoder_sender_hidden = gather_rows(&full_hidden, &graphs.decoder.senders)?;
        let decoder_receiver_data = gather_rows(&projected_state, &graphs.decoder.receivers)?;
        let decoder_edge_input = concat_tensors(&[
            &graphs.decoder.edge_tensors.local_coords_row,
            &decoder_sender_hidden,
            &decoder_receiver_data,
        ])?;
        let decoder_edge_hidden = self.decoder_edge_mlp.forward(&decoder_edge_input)?;
        let decoder_agg = aggregate_receivers(
            &decoder_edge_hidden,
            &graphs.decoder.receivers,
            graphs.decoder.n_nodes,
        )?;
        let decoder_node_input = concat_tensors(&[
            &projected_state,
            &elementwise_mul(&decoder_agg, &graphs.decoder.node_tensors.inv_n_senders)?,
        ])?;
        let decoder_hidden = self.decoder_node_mlp.forward(&decoder_node_input)?;
        self.output_projection.forward(&decoder_hidden)
    }
}

fn concat_tensors(tensors: &[&Tensor]) -> Result<Tensor> {
    let rows = tensors
        .first()
        .map_or(0, |tensor| tensor.dims2().map_or(0, |dims| dims.0));
    let mut pieces = Vec::with_capacity(tensors.len());
    let mut total_width = 0_usize;
    for tensor in tensors {
        let (tensor_rows, width) = tensor.dims2()?;
        if tensor_rows != rows {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "concat_tensors".to_owned(),
                expected: rows.to_string(),
                actual: tensor_rows.to_string(),
            });
        }
        total_width += width;
        pieces.push(tensor.to_vec2::<f32>()?);
    }

    let mut values = Vec::with_capacity(rows * total_width);
    for row in 0..rows {
        for piece in &pieces {
            values.extend_from_slice(&piece[row]);
        }
    }
    Ok(Tensor::from_vec(values, (rows, total_width), &Device::Cpu)?)
}

fn add_tensors(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let left = lhs.to_vec2::<f32>()?;
    let right = rhs.to_vec2::<f32>()?;
    if left.len() != right.len()
        || left.first().map_or(0, Vec::len) != right.first().map_or(0, Vec::len)
    {
        return Err(WeatherGraphError::ShapeMismatch {
            name: "add_tensors".to_owned(),
            expected: format!("{:?}", lhs.dims()),
            actual: format!("{:?}", rhs.dims()),
        });
    }
    let width = left.first().map_or(0, Vec::len);
    let values = left
        .into_iter()
        .zip(right)
        .flat_map(|(lhs_row, rhs_row)| lhs_row.into_iter().zip(rhs_row).map(|(l, r)| l + r))
        .collect::<Vec<_>>();
    let batch = values.len() / width;
    Ok(Tensor::from_vec(values, (batch, width), &Device::Cpu)?)
}

fn elementwise_mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let left = lhs.to_vec2::<f32>()?;
    let right = rhs.to_vec2::<f32>()?;
    if left.len() != right.len() {
        return Err(WeatherGraphError::ShapeMismatch {
            name: "elementwise_mul".to_owned(),
            expected: left.len().to_string(),
            actual: right.len().to_string(),
        });
    }
    let right_width = right.first().map_or(0, Vec::len);
    let left_width = left.first().map_or(0, Vec::len);
    if right_width != 1 && right_width != left_width {
        return Err(WeatherGraphError::ShapeMismatch {
            name: "elementwise_mul".to_owned(),
            expected: format!("rhs width 1 or {left_width}"),
            actual: right_width.to_string(),
        });
    }
    let values = left
        .into_iter()
        .zip(right)
        .flat_map(|(lhs_row, rhs_row)| {
            lhs_row.into_iter().enumerate().map(move |(index, value)| {
                let scale = if rhs_row.len() == 1 {
                    rhs_row[0]
                } else {
                    rhs_row[index]
                };
                value * scale
            })
        })
        .collect::<Vec<_>>();
    let batch = values.len() / left_width;
    Ok(Tensor::from_vec(values, (batch, left_width), &Device::Cpu)?)
}

fn slice_rows(tensor: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    let values = tensor.to_vec2::<f32>()?;
    let width = values.first().map_or(0, Vec::len);
    let slice = values
        .into_iter()
        .skip(start)
        .take(len)
        .flatten()
        .collect::<Vec<_>>();
    Ok(Tensor::from_vec(slice, (len, width), &Device::Cpu)?)
}

fn pad_h3_hidden(
    h3_hidden: &Tensor,
    total_nodes: usize,
    era5_nodes: usize,
    hidden_dim: usize,
) -> Result<Tensor> {
    let mut values = vec![0.0_f32; total_nodes * hidden_dim];
    let h3_values = h3_hidden.to_vec2::<f32>()?;
    for (row_index, row) in h3_values.iter().enumerate() {
        let offset = (era5_nodes + row_index) * hidden_dim;
        values[offset..offset + hidden_dim].copy_from_slice(row);
    }
    Ok(Tensor::from_vec(
        values,
        (total_nodes, hidden_dim),
        &Device::Cpu,
    )?)
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

fn load_mlp_optional(
    prefixes: &[&str],
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    hidden_dim: usize,
    with_layer_norm: bool,
    device: &Device,
) -> Result<Option<Mlp>> {
    let has_any_prefix = prefixes
        .iter()
        .any(|prefix| has_any_tensor_key(prefix, tensors));
    if !has_any_prefix {
        return Ok(None);
    }
    load_mlp_with_aliases(
        prefixes,
        tensors,
        hidden_dim,
        hidden_dim,
        with_layer_norm,
        device,
    )
    .map(Some)
}

fn load_mlp_with_aliases(
    prefixes: &[&str],
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    input_dim: usize,
    hidden_dim: usize,
    with_layer_norm: bool,
    device: &Device,
) -> Result<Mlp> {
    let layer0 = LinearLayer {
        weight: load_tensor_any(
            tensors,
            &candidate_param_keys(prefixes, "layers.0", "weight"),
            device,
        )?,
        bias: load_tensor_any_optional(
            tensors,
            &candidate_param_keys(prefixes, "layers.0", "bias"),
            device,
        )?,
    };
    let layer1 = LinearLayer {
        weight: load_tensor_any(
            tensors,
            &candidate_param_keys(prefixes, "layers.1", "weight"),
            device,
        )?,
        bias: load_tensor_any_optional(
            tensors,
            &candidate_param_keys(prefixes, "layers.1", "bias"),
            device,
        )?,
    };

    if layer0.weight.dims2()? != (hidden_dim, input_dim) {
        return Err(WeatherGraphError::ShapeMismatch {
            name: format!("{}.layers.0.weight", prefixes[0]),
            expected: format!("[{hidden_dim}, {input_dim}]"),
            actual: format!("{:?}", layer0.weight.dims()),
        });
    }
    if layer1.weight.dims2()? != (hidden_dim, hidden_dim) {
        return Err(WeatherGraphError::ShapeMismatch {
            name: format!("{}.layers.1.weight", prefixes[0]),
            expected: format!("[{hidden_dim}, {hidden_dim}]"),
            actual: format!("{:?}", layer1.weight.dims()),
        });
    }

    let layer_norm = if with_layer_norm {
        Some(LayerNorm {
            weight: load_tensor_any(
                tensors,
                &candidate_layer_norm_keys(prefixes, "weight"),
                device,
            )?,
            bias: load_tensor_any(
                tensors,
                &candidate_layer_norm_keys(prefixes, "bias"),
                device,
            )?,
            eps: 1.0e-5_f32,
        })
    } else {
        None
    };

    Ok(Mlp {
        layers: vec![layer0, layer1],
        layer_norm,
    })
}

fn load_linear_or_identity(
    prefix: &str,
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    input_dim: usize,
    output_dim: usize,
    device: &Device,
) -> Result<LinearLayer> {
    if let Some(weight) = load_tensor_any_optional(
        tensors,
        &[format!("{prefix}.weight"), format!("{prefix}.w")],
        device,
    )? {
        let bias = load_tensor_any_optional(
            tensors,
            &[format!("{prefix}.bias"), format!("{prefix}.b")],
            device,
        )?;
        return Ok(LinearLayer { weight, bias });
    }

    linear_identity(input_dim, output_dim, device)
}

fn load_tensor_any(
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    keys: &[String],
    device: &Device,
) -> Result<Tensor> {
    load_tensor_any_optional(tensors, keys, device)?.ok_or_else(|| {
        WeatherGraphError::MissingArtifact {
            name: keys.join(" | "),
            path: Path::new("weights.safetensors").to_path_buf(),
        }
    })
}

fn load_tensor_any_optional(
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    keys: &[String],
    device: &Device,
) -> Result<Option<Tensor>> {
    for key in keys {
        if let Some((shape, values)) = tensors.get(key) {
            let tensor = Tensor::from_vec(values.clone(), shape.as_slice(), device)?;
            return Ok(Some(tensor));
        }
    }
    Ok(None)
}

fn candidate_param_keys(prefixes: &[&str], layer: &str, param: &str) -> Vec<String> {
    let suffixes = match param {
        "weight" => ["weight", "w"],
        "bias" => ["bias", "b"],
        _ => [param, param],
    };
    prefixes
        .iter()
        .flat_map(|prefix| {
            [
                format!("{prefix}.{layer}.{}", suffixes[0]),
                format!("{prefix}.{layer}.{}", suffixes[1]),
            ]
        })
        .collect()
}

fn candidate_layer_norm_keys(prefixes: &[&str], param: &str) -> Vec<String> {
    let suffixes = match param {
        "weight" => ["layer_norm.weight", "layer_norm.scale"],
        "bias" => ["layer_norm.bias", "layer_norm.offset"],
        _ => [param, param],
    };
    prefixes
        .iter()
        .flat_map(|prefix| {
            [
                format!("{prefix}.{}", suffixes[0]),
                format!("{prefix}.{}", suffixes[1]),
            ]
        })
        .collect()
}

fn has_any_tensor_key(prefix: &str, tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>) -> bool {
    tensors.contains_key(&format!("{prefix}.layers.0.weight"))
        || tensors.contains_key(&format!("{prefix}.layers.0.w"))
        || tensors.contains_key(&format!("{prefix}.w"))
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
            input_channels: 2,
            output_channels: 2,
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
            input_channels: 2,
            output_channels: 2,
            hidden_dim: 2,
            processor_blocks: 1,
            use_layer_norm: true,
        };
        let mut tensors = HashMap::new();
        tensors.insert(
            "input_projection.weight".to_owned(),
            (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        );
        tensors.insert(
            "input_projection.bias".to_owned(),
            (vec![2], vec![0.0, 0.0]),
        );
        tensors.insert(
            "output_projection.weight".to_owned(),
            (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        );
        tensors.insert(
            "output_projection.bias".to_owned(),
            (vec![2], vec![0.0, 0.0]),
        );
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

    #[test]
    fn gnn_loads_alias_style_weight_map() {
        let device = Device::Cpu;
        let config = ModelConfig {
            input_channels: 2,
            output_channels: 2,
            hidden_dim: 2,
            processor_blocks: 1,
            use_layer_norm: true,
        };
        let mut tensors = HashMap::new();
        tensors.insert(
            "input_projection.w".to_owned(),
            (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        );
        tensors.insert("input_projection.b".to_owned(), (vec![2], vec![0.0, 0.0]));
        tensors.insert(
            "output_projection.w".to_owned(),
            (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        );
        tensors.insert("output_projection.b".to_owned(), (vec![2], vec![0.0, 0.0]));

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
                    format!("{prefix}.{layer}.w"),
                    (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
                );
                tensors.insert(format!("{prefix}.{layer}.b"), (vec![2], vec![0.0, 0.0]));
            }
        }

        for prefix in [
            "encoder_edge_mlp",
            "encoder_node_mlp",
            "processor_edge_mlp",
            "processor_node_mlp",
            "decoder_edge_mlp",
        ] {
            tensors.insert(
                format!("{prefix}.layer_norm.scale"),
                (vec![2], vec![1.0, 1.0]),
            );
            tensors.insert(
                format!("{prefix}.layer_norm.offset"),
                (vec![2], vec![0.0, 0.0]),
            );
        }

        tensors.insert(
            "net_edges.layers.0.w".to_owned(),
            (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        );
        tensors.insert("net_edges.layers.0.b".to_owned(), (vec![2], vec![0.0, 0.0]));
        tensors.insert(
            "net_edges.layers.1.w".to_owned(),
            (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        );
        tensors.insert("net_edges.layers.1.b".to_owned(), (vec![2], vec![0.0, 0.0]));
        tensors.insert(
            "net_edges.layer_norm.scale".to_owned(),
            (vec![2], vec![1.0, 1.0]),
        );
        tensors.insert(
            "net_edges.layer_norm.offset".to_owned(),
            (vec![2], vec![0.0, 0.0]),
        );

        let model =
            KeislerGnn::from_weight_map(&tensors, &config, &device).expect("alias weight map");
        let input = Tensor::from_vec(vec![1.0_f32, 2.0], (1, 2), &device).expect("input");
        let output = model.one_step(&input).expect("one step");
        assert_eq!(output.dims2().expect("dims"), (1, 2));
    }
}
