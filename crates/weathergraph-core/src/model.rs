use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use safetensors::{Dtype, SafeTensors};
use serde::Serialize;

use crate::config::ModelConfig;
use crate::error::{Result, WeatherGraphError};
use crate::graph::GraphSet;
use crate::solar::SOLAR_TIME_SHIFTS_HOURS;
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WeightMatch {
    pub canonical_key: String,
    pub matched_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WeightInspectionReport {
    pub available_keys: Vec<String>,
    pub matched_required: Vec<WeightMatch>,
    pub matched_optional: Vec<WeightMatch>,
    pub missing_required: Vec<String>,
    pub missing_optional: Vec<String>,
    pub dtype_mismatches: Vec<WeightDtypeMismatch>,
    pub shape_mismatches: Vec<WeightShapeMismatch>,
    pub unused_keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WeightDtypeMismatch {
    pub canonical_key: String,
    pub matched_key: String,
    pub expected_dtype: String,
    pub actual_dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WeightShapeMismatch {
    pub canonical_key: String,
    pub matched_key: String,
    pub expected_shape: Vec<usize>,
    pub actual_shape: Vec<usize>,
}

impl WeightInspectionReport {
    pub fn required_coverage(&self) -> String {
        format!(
            "{}/{}",
            self.matched_required.len(),
            self.matched_required.len() + self.missing_required.len()
        )
    }
}

impl KeislerGnn {
    pub fn placeholder(config: &ModelConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            input_projection: linear_identity(
                config.input_channels,
                config.input_channels,
                device,
            )?,
            encoder_edge_mlp: identity_mlp(
                encoder_edge_input_dim(config),
                config.hidden_dim,
                config.hidden_dim,
                config.n_mlp_layers_encoder,
                config.use_layer_norm,
                device,
            )?,
            encoder_node_mlp: identity_mlp(
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim,
                config.n_mlp_layers_encoder,
                config.use_layer_norm,
                device,
            )?,
            processor_edge_init_mlp: identity_mlp(
                2,
                config.hidden_dim,
                config.hidden_dim,
                0,
                config.use_layer_norm,
                device,
            )?,
            processor_edge_mlp: identity_mlp(
                config.hidden_dim * 3,
                config.hidden_dim,
                config.hidden_dim,
                config.n_mlp_layers_processor,
                config.use_layer_norm,
                device,
            )?,
            processor_node_mlp: identity_mlp(
                (config.hidden_dim * 3) + 4,
                config.hidden_dim,
                config.hidden_dim,
                config.n_mlp_layers_processor,
                config.use_layer_norm,
                device,
            )?,
            decoder_edge_mlp: identity_mlp(
                2 + config.hidden_dim + config.input_channels,
                config.hidden_dim,
                config.hidden_dim,
                config.n_mlp_layers_decoder,
                config.use_layer_norm,
                device,
            )?,
            decoder_node_mlp: identity_mlp(
                config.input_channels + config.hidden_dim,
                config.hidden_dim,
                config.output_channels,
                config.n_mlp_layers_decoder,
                false,
                device,
            )?,
            output_projection: linear_identity(
                config.output_channels,
                config.output_channels,
                device,
            )?,
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
                if view.dtype() != Dtype::F32 {
                    return Err(WeatherGraphError::UnsupportedDtype {
                        name: (*name).to_owned(),
                        dtype: format!("{:?}", view.dtype()),
                    });
                }
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

    pub fn inspect_safetensors(
        path: impl AsRef<Path>,
        config: &ModelConfig,
    ) -> Result<WeightInspectionReport> {
        let bytes = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let tensor_metadata = tensors
            .names()
            .iter()
            .map(|name| {
                let view = tensors.tensor(name)?;
                Ok((
                    (*name).to_owned(),
                    TensorMetadata {
                        shape: view.shape().to_vec(),
                        dtype: format!("{:?}", view.dtype()),
                    },
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(inspect_weight_metadata(&tensor_metadata, config))
    }

    pub fn from_weight_map(
        tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        config: &ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        let processor_edge_mlp = load_mlp_with_aliases(
            &["processor_edge_mlp"],
            tensors,
            config.hidden_dim * 3,
            config.hidden_dim,
            config.hidden_dim,
            config.use_layer_norm,
            config.n_mlp_layers_processor,
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
                encoder_edge_input_dim(config),
                config.hidden_dim,
                config.hidden_dim,
                config.use_layer_norm,
                config.n_mlp_layers_encoder,
                device,
            )?,
            encoder_node_mlp: load_mlp_with_aliases(
                &["encoder_node_mlp"],
                tensors,
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim,
                config.use_layer_norm,
                config.n_mlp_layers_encoder,
                device,
            )?,
            processor_edge_init_mlp: load_mlp_optional(
                &[
                    "processor_edge_init_mlp",
                    "processor_edge_mlp_init",
                    "net_edges",
                ],
                tensors,
                2,
                config.hidden_dim,
                config.hidden_dim,
                true,
                0,
                device,
            )?
            .unwrap_or_else(|| processor_edge_mlp.clone()),
            processor_edge_mlp,
            processor_node_mlp: load_mlp_with_aliases(
                &["processor_node_mlp"],
                tensors,
                (config.hidden_dim * 3) + 4,
                config.hidden_dim,
                config.hidden_dim,
                config.use_layer_norm,
                config.n_mlp_layers_processor,
                device,
            )?,
            decoder_edge_mlp: load_mlp_with_aliases(
                &["decoder_edge_mlp"],
                tensors,
                2 + config.hidden_dim + config.input_channels,
                config.hidden_dim,
                config.hidden_dim,
                config.use_layer_norm,
                config.n_mlp_layers_decoder,
                device,
            )?,
            decoder_node_mlp: load_mlp_with_aliases(
                &["decoder_node_mlp"],
                tensors,
                config.input_channels + config.hidden_dim,
                config.hidden_dim,
                config.output_channels,
                false,
                config.n_mlp_layers_decoder,
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
        Ok(state.clone())
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
        let sender_state = gather_rows(state, &graphs.encoder.senders)?;
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
        let decoder_receiver_data = gather_rows(state, &graphs.decoder.receivers)?;
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
            state,
            &elementwise_mul(&decoder_agg, &graphs.decoder.node_tensors.inv_n_senders)?,
        ])?;
        let decoder_change = self.decoder_node_mlp.forward(&decoder_node_input)?;
        add_tensors(state, &decoder_change)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorMetadata {
    shape: Vec<usize>,
    dtype: String,
}

#[cfg(test)]
fn inspect_weight_keys(keys: &[String], config: &ModelConfig) -> WeightInspectionReport {
    let metadata = keys
        .iter()
        .map(|key| {
            (
                key.clone(),
                TensorMetadata {
                    shape: Vec::new(),
                    dtype: "unknown".to_owned(),
                },
            )
        })
        .collect::<HashMap<_, _>>();
    inspect_weight_metadata(&metadata, config)
}

fn inspect_weight_metadata(
    metadata: &HashMap<String, TensorMetadata>,
    config: &ModelConfig,
) -> WeightInspectionReport {
    let mut available_keys = metadata.keys().cloned().collect::<Vec<_>>();
    available_keys.sort_unstable();

    let mut matched_required = Vec::new();
    let mut matched_optional = Vec::new();
    let mut missing_required = Vec::new();
    let mut missing_optional = Vec::new();
    let mut dtype_mismatches = Vec::new();
    let mut shape_mismatches = Vec::new();

    for expected in expected_weight_entries(config) {
        if let Some(matched_key) = expected
            .candidate_keys
            .iter()
            .find(|candidate| metadata.contains_key(*candidate))
        {
            let tensor_metadata = metadata
                .get(matched_key)
                .expect("matched key must have metadata");
            let canonical_key = expected.canonical_key.clone();
            let matched = WeightMatch {
                canonical_key: canonical_key.clone(),
                matched_key: matched_key.clone(),
            };
            if expected.required {
                matched_required.push(matched);
            } else {
                matched_optional.push(matched);
            }

            if tensor_metadata.dtype != "unknown"
                && tensor_metadata.dtype != expected.expected_dtype
            {
                dtype_mismatches.push(WeightDtypeMismatch {
                    canonical_key: canonical_key.clone(),
                    matched_key: matched_key.clone(),
                    expected_dtype: expected.expected_dtype.clone(),
                    actual_dtype: tensor_metadata.dtype.clone(),
                });
            }
            if !tensor_metadata.shape.is_empty() && tensor_metadata.shape != expected.expected_shape
            {
                shape_mismatches.push(WeightShapeMismatch {
                    canonical_key,
                    matched_key: matched_key.clone(),
                    expected_shape: expected.expected_shape.clone(),
                    actual_shape: tensor_metadata.shape.clone(),
                });
            }
        } else if expected.required {
            missing_required.push(expected.canonical_key);
        } else {
            missing_optional.push(expected.canonical_key);
        }
    }

    let used_keys = matched_required
        .iter()
        .chain(&matched_optional)
        .map(|entry| entry.matched_key.clone())
        .collect::<Vec<_>>();
    let unused_keys = available_keys
        .iter()
        .filter(|key| !used_keys.contains(*key))
        .cloned()
        .collect();

    WeightInspectionReport {
        available_keys,
        matched_required,
        matched_optional,
        missing_required,
        missing_optional,
        dtype_mismatches,
        shape_mismatches,
        unused_keys,
    }
}

#[derive(Debug, Clone)]
struct ExpectedWeightEntry {
    canonical_key: String,
    candidate_keys: Vec<String>,
    expected_dtype: String,
    expected_shape: Vec<usize>,
    required: bool,
}

fn encoder_edge_input_dim(config: &ModelConfig) -> usize {
    let mut width = 2 + config.input_channels + SOLAR_TIME_SHIFTS_HOURS.len() + 2;
    if config.use_lat {
        width += 2;
    }
    if config.use_lon {
        width += 2;
    }
    if config.use_doy {
        width += 1;
    }
    width
}

fn encoder_like_input_dim(prefix: &str, config: &ModelConfig) -> usize {
    match prefix {
        "encoder_edge_mlp" => encoder_edge_input_dim(config),
        "processor_edge_mlp" => config.hidden_dim * 3,
        "processor_node_mlp" => (config.hidden_dim * 3) + 4,
        "decoder_edge_mlp" => 2 + config.hidden_dim + config.input_channels,
        _ => config.hidden_dim,
    }
}

fn expected_weight_entries(config: &ModelConfig) -> Vec<ExpectedWeightEntry> {
    let mut entries = Vec::new();
    entries.extend(expected_linear_entries(
        "input_projection",
        config.input_channels,
        config.hidden_dim,
        false,
    ));
    entries.extend(expected_linear_entries(
        "output_projection",
        config.hidden_dim,
        config.output_channels,
        false,
    ));

    for prefix in [
        "encoder_edge_mlp",
        "encoder_node_mlp",
        "processor_edge_mlp",
        "processor_node_mlp",
        "decoder_edge_mlp",
    ] {
        entries.extend(expected_mlp_entries(
            prefix,
            encoder_like_input_dim(prefix, config),
            config.hidden_dim,
            config.hidden_dim,
            config.use_layer_norm,
            true,
            match prefix {
                "encoder_edge_mlp" | "encoder_node_mlp" => config.n_mlp_layers_encoder,
                "processor_edge_mlp" | "processor_node_mlp" => config.n_mlp_layers_processor,
                "decoder_edge_mlp" => config.n_mlp_layers_decoder,
                _ => 0,
            },
            &[prefix],
        ));
    }
    entries.extend(expected_mlp_entries(
        "decoder_node_mlp",
        config.input_channels + config.hidden_dim,
        config.hidden_dim,
        config.output_channels,
        false,
        true,
        config.n_mlp_layers_decoder,
        &["decoder_node_mlp"],
    ));
    entries.extend(expected_mlp_entries(
        "processor_edge_init_mlp",
        2,
        config.hidden_dim,
        config.hidden_dim,
        true,
        false,
        0,
        &[
            "processor_edge_init_mlp",
            "processor_edge_mlp_init",
            "net_edges",
        ],
    ));
    entries
}

fn expected_linear_entries(
    prefix: &str,
    input_dim: usize,
    output_dim: usize,
    required: bool,
) -> Vec<ExpectedWeightEntry> {
    vec![
        ExpectedWeightEntry {
            canonical_key: format!("{prefix}.weight"),
            candidate_keys: vec![format!("{prefix}.weight"), format!("{prefix}.w")],
            expected_dtype: "F32".to_owned(),
            expected_shape: vec![output_dim, input_dim],
            required,
        },
        ExpectedWeightEntry {
            canonical_key: format!("{prefix}.bias"),
            candidate_keys: vec![format!("{prefix}.bias"), format!("{prefix}.b")],
            expected_dtype: "F32".to_owned(),
            expected_shape: vec![output_dim],
            required: false,
        },
    ]
}

#[allow(clippy::too_many_arguments)]
fn expected_mlp_entries(
    canonical_prefix: &str,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    with_layer_norm: bool,
    required: bool,
    hidden_layers: usize,
    prefixes: &[&str],
) -> Vec<ExpectedWeightEntry> {
    let total_layers = hidden_layers + 1;
    let mut entries = Vec::with_capacity(total_layers * 2);
    for layer_index in 0..total_layers {
        let is_first = layer_index == 0;
        let is_last = layer_index + 1 == total_layers;
        let this_input_dim = if is_first { input_dim } else { hidden_dim };
        let this_output_dim = if is_last { output_dim } else { hidden_dim };
        entries.push(ExpectedWeightEntry {
            canonical_key: format!("{canonical_prefix}.layers.{layer_index}.weight"),
            candidate_keys: candidate_param_keys(
                prefixes,
                &format!("layers.{layer_index}"),
                "weight",
            ),
            expected_dtype: "F32".to_owned(),
            expected_shape: vec![this_output_dim, this_input_dim],
            required,
        });
        entries.push(ExpectedWeightEntry {
            canonical_key: format!("{canonical_prefix}.layers.{layer_index}.bias"),
            candidate_keys: candidate_param_keys(
                prefixes,
                &format!("layers.{layer_index}"),
                "bias",
            ),
            expected_dtype: "F32".to_owned(),
            expected_shape: vec![this_output_dim],
            required: false,
        });
    }

    if with_layer_norm {
        entries.push(ExpectedWeightEntry {
            canonical_key: format!("{canonical_prefix}.layer_norm.weight"),
            candidate_keys: candidate_layer_norm_keys(prefixes, "weight"),
            expected_dtype: "F32".to_owned(),
            expected_shape: vec![hidden_dim],
            required,
        });
        entries.push(ExpectedWeightEntry {
            canonical_key: format!("{canonical_prefix}.layer_norm.bias"),
            candidate_keys: candidate_layer_norm_keys(prefixes, "bias"),
            expected_dtype: "F32".to_owned(),
            expected_shape: vec![hidden_dim],
            required,
        });
    }

    entries
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

fn identity_mlp(
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    hidden_layers: usize,
    with_layer_norm: bool,
    device: &Device,
) -> Result<Mlp> {
    let total_layers = hidden_layers + 1;
    let mut layers = Vec::with_capacity(total_layers);
    for layer_index in 0..total_layers {
        let layer_input_dim = if layer_index == 0 {
            input_dim
        } else {
            hidden_dim
        };
        let layer_output_dim = if layer_index + 1 == total_layers {
            output_dim
        } else {
            hidden_dim
        };
        layers.push(linear_identity(layer_input_dim, layer_output_dim, device)?);
    }

    Ok(Mlp {
        layers,
        layer_norm: with_layer_norm
            .then(|| layer_norm_identity(output_dim, device))
            .transpose()?,
    })
}

fn layer_norm_identity(width: usize, device: &Device) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: Tensor::ones(width, candle_core::DType::F32, device)?,
        bias: Tensor::zeros(width, candle_core::DType::F32, device)?,
        eps: 1.0e-5_f32,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_mlp_optional(
    prefixes: &[&str],
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    with_layer_norm: bool,
    hidden_layers: usize,
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
        input_dim,
        hidden_dim,
        output_dim,
        with_layer_norm,
        hidden_layers,
        device,
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
fn load_mlp_with_aliases(
    prefixes: &[&str],
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    with_layer_norm: bool,
    hidden_layers: usize,
    device: &Device,
) -> Result<Mlp> {
    let total_layers = hidden_layers + 1;
    let mut layers = Vec::with_capacity(total_layers);
    for layer_index in 0..total_layers {
        let layer_name = format!("layers.{layer_index}");
        let layer = LinearLayer {
            weight: load_tensor_any(
                tensors,
                &candidate_param_keys(prefixes, &layer_name, "weight"),
                device,
            )?,
            bias: load_tensor_any_optional(
                tensors,
                &candidate_param_keys(prefixes, &layer_name, "bias"),
                device,
            )?,
        };
        let expected_input_dim = if layer_index == 0 {
            input_dim
        } else {
            hidden_dim
        };
        let expected_output_dim = if layer_index + 1 == total_layers {
            output_dim
        } else {
            hidden_dim
        };
        if layer.weight.dims2()? != (expected_output_dim, expected_input_dim) {
            return Err(WeatherGraphError::ShapeMismatch {
                name: format!("{}.layers.{layer_index}.weight", prefixes[0]),
                expected: format!("[{expected_output_dim}, {expected_input_dim}]"),
                actual: format!("{:?}", layer.weight.dims()),
            });
        }
        layers.push(layer);
    }

    let layer_norm = if with_layer_norm {
        let weight = load_tensor_any(
            tensors,
            &candidate_layer_norm_keys(prefixes, "weight"),
            device,
        )?;
        let bias = load_tensor_any(
            tensors,
            &candidate_layer_norm_keys(prefixes, "bias"),
            device,
        )?;
        if weight.dims1()? != hidden_dim {
            return Err(WeatherGraphError::ShapeMismatch {
                name: format!("{}.layer_norm.weight", prefixes[0]),
                expected: format!("[{hidden_dim}]"),
                actual: format!("{:?}", weight.dims()),
            });
        }
        if bias.dims1()? != hidden_dim {
            return Err(WeatherGraphError::ShapeMismatch {
                name: format!("{}.layer_norm.bias", prefixes[0]),
                expected: format!("[{hidden_dim}]"),
                actual: format!("{:?}", bias.dims()),
            });
        }
        Some(LayerNorm {
            weight,
            bias,
            eps: 1.0e-5_f32,
        })
    } else {
        None
    };

    Ok(Mlp { layers, layer_norm })
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
        if weight.dims2()? != (output_dim, input_dim) {
            return Err(WeatherGraphError::ShapeMismatch {
                name: format!("{prefix}.weight"),
                expected: format!("[{output_dim}, {input_dim}]"),
                actual: format!("{:?}", weight.dims()),
            });
        }
        let bias = load_tensor_any_optional(
            tensors,
            &[format!("{prefix}.bias"), format!("{prefix}.b")],
            device,
        )?;
        if let Some(bias_tensor) = &bias {
            let actual = bias_tensor.dims1()?;
            if actual != output_dim {
                return Err(WeatherGraphError::ShapeMismatch {
                    name: format!("{prefix}.bias"),
                    expected: format!("[{output_dim}]"),
                    actual: format!("[{actual}]"),
                });
            }
        }
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

    use super::{
        KeislerGnn, Mlp, TensorMetadata, WeightInspectionReport, encoder_edge_input_dim,
        inspect_weight_keys, inspect_weight_metadata, layer_norm_identity, linear_identity,
    };
    use crate::config::ModelConfig;

    fn test_config() -> ModelConfig {
        ModelConfig {
            input_channels: 2,
            output_channels: 2,
            hidden_dim: 2,
            processor_blocks: 1,
            use_layer_norm: true,
            ..ModelConfig::default()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_mlp(
        tensors: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
        prefix: &str,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        hidden_layers: usize,
        with_layer_norm: bool,
        alias_style: bool,
    ) {
        let total_layers = hidden_layers + 1;
        for layer_index in 0..total_layers {
            let layer_input = if layer_index == 0 {
                input_dim
            } else {
                hidden_dim
            };
            let layer_output = if layer_index + 1 == total_layers {
                output_dim
            } else {
                hidden_dim
            };
            let suffix_w = if alias_style { "w" } else { "weight" };
            let suffix_b = if alias_style { "b" } else { "bias" };
            tensors.insert(
                format!("{prefix}.layers.{layer_index}.{suffix_w}"),
                (
                    vec![layer_output, layer_input],
                    vec![0.0; layer_output * layer_input],
                ),
            );
            tensors.insert(
                format!("{prefix}.layers.{layer_index}.{suffix_b}"),
                (vec![layer_output], vec![0.0; layer_output]),
            );
        }

        if with_layer_norm {
            let suffix_w = if alias_style { "scale" } else { "weight" };
            let suffix_b = if alias_style { "offset" } else { "bias" };
            tensors.insert(
                format!("{prefix}.layer_norm.{suffix_w}"),
                (vec![output_dim], vec![1.0; output_dim]),
            );
            tensors.insert(
                format!("{prefix}.layer_norm.{suffix_b}"),
                (vec![output_dim], vec![0.0; output_dim]),
            );
        }
    }

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
            processor_blocks: 2,
            ..test_config()
        };
        let model = KeislerGnn::placeholder(&config, &device).expect("placeholder");
        let input = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2), &device).expect("input");
        let output = model.one_step(&input).expect("one step");
        assert_eq!(output.dims2().expect("dims"), (2, 2));
    }

    #[test]
    fn gnn_loads_fake_weight_map() {
        let device = Device::Cpu;
        let config = test_config();
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
        insert_mlp(
            &mut tensors,
            "encoder_edge_mlp",
            encoder_edge_input_dim(&config),
            config.hidden_dim,
            config.hidden_dim,
            config.n_mlp_layers_encoder,
            true,
            false,
        );
        insert_mlp(
            &mut tensors,
            "encoder_node_mlp",
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.n_mlp_layers_encoder,
            true,
            false,
        );
        insert_mlp(
            &mut tensors,
            "processor_edge_mlp",
            config.hidden_dim * 3,
            config.hidden_dim,
            config.hidden_dim,
            config.n_mlp_layers_processor,
            true,
            false,
        );
        insert_mlp(
            &mut tensors,
            "processor_node_mlp",
            (config.hidden_dim * 3) + 4,
            config.hidden_dim,
            config.hidden_dim,
            config.n_mlp_layers_processor,
            true,
            false,
        );
        insert_mlp(
            &mut tensors,
            "decoder_edge_mlp",
            2 + config.hidden_dim + config.input_channels,
            config.hidden_dim,
            config.hidden_dim,
            config.n_mlp_layers_decoder,
            true,
            false,
        );
        insert_mlp(
            &mut tensors,
            "decoder_node_mlp",
            config.input_channels + config.hidden_dim,
            config.hidden_dim,
            config.output_channels,
            config.n_mlp_layers_decoder,
            false,
            false,
        );

        let model = KeislerGnn::from_weight_map(&tensors, &config, &device).expect("weight map");
        let input = Tensor::from_vec(vec![1.0_f32, 2.0], (1, 2), &device).expect("input");
        let output = model.one_step(&input).expect("one step");
        assert_eq!(output.dims2().expect("dims"), (1, 2));
    }

    #[test]
    fn gnn_loads_alias_style_weight_map() {
        let device = Device::Cpu;
        let config = test_config();
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

        insert_mlp(
            &mut tensors,
            "encoder_edge_mlp",
            encoder_edge_input_dim(&config),
            config.hidden_dim,
            config.hidden_dim,
            config.n_mlp_layers_encoder,
            true,
            true,
        );
        insert_mlp(
            &mut tensors,
            "encoder_node_mlp",
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.n_mlp_layers_encoder,
            true,
            true,
        );
        insert_mlp(
            &mut tensors,
            "processor_edge_mlp",
            config.hidden_dim * 3,
            config.hidden_dim,
            config.hidden_dim,
            config.n_mlp_layers_processor,
            true,
            true,
        );
        insert_mlp(
            &mut tensors,
            "processor_node_mlp",
            (config.hidden_dim * 3) + 4,
            config.hidden_dim,
            config.hidden_dim,
            config.n_mlp_layers_processor,
            true,
            true,
        );
        insert_mlp(
            &mut tensors,
            "decoder_edge_mlp",
            2 + config.hidden_dim + config.input_channels,
            config.hidden_dim,
            config.hidden_dim,
            config.n_mlp_layers_decoder,
            true,
            true,
        );
        insert_mlp(
            &mut tensors,
            "decoder_node_mlp",
            config.input_channels + config.hidden_dim,
            config.hidden_dim,
            config.output_channels,
            config.n_mlp_layers_decoder,
            false,
            true,
        );
        insert_mlp(
            &mut tensors,
            "net_edges",
            2,
            config.hidden_dim,
            config.hidden_dim,
            0,
            true,
            true,
        );

        let model =
            KeislerGnn::from_weight_map(&tensors, &config, &device).expect("alias weight map");
        let input = Tensor::from_vec(vec![1.0_f32, 2.0], (1, 2), &device).expect("input");
        let output = model.one_step(&input).expect("one step");
        assert_eq!(output.dims2().expect("dims"), (1, 2));
    }

    #[test]
    fn inspect_weights_reports_alias_hits_and_missing_keys() {
        let config = ModelConfig { ..test_config() };
        let report: WeightInspectionReport = inspect_weight_keys(
            &[
                "encoder_edge_mlp.layers.0.w".to_owned(),
                "encoder_edge_mlp.layers.1.w".to_owned(),
                "encoder_edge_mlp.layer_norm.scale".to_owned(),
                "encoder_edge_mlp.layer_norm.offset".to_owned(),
                "unused.tensor".to_owned(),
            ],
            &config,
        );

        assert!(
            report
                .matched_required
                .iter()
                .any(
                    |entry| entry.canonical_key == "encoder_edge_mlp.layers.0.weight"
                        && entry.matched_key == "encoder_edge_mlp.layers.0.w"
                )
        );
        assert!(
            report
                .missing_required
                .contains(&"encoder_node_mlp.layers.0.weight".to_owned())
        );
        assert!(report.shape_mismatches.is_empty());
        assert!(report.unused_keys.contains(&"unused.tensor".to_owned()));
    }

    #[test]
    fn inspect_weights_reports_shape_mismatches() {
        let config = ModelConfig {
            hidden_dim: 3,
            ..test_config()
        };
        let report = inspect_weight_metadata(
            &HashMap::from([(
                "encoder_edge_mlp.layers.0.w".to_owned(),
                TensorMetadata {
                    shape: vec![2, 2],
                    dtype: "F32".to_owned(),
                },
            )]),
            &config,
        );

        assert!(report.shape_mismatches.iter().any(|mismatch| {
            mismatch.canonical_key == "encoder_edge_mlp.layers.0.weight"
                && mismatch.expected_shape == vec![3, encoder_edge_input_dim(&config)]
                && mismatch.actual_shape == vec![2, 2]
        }));
    }
}
