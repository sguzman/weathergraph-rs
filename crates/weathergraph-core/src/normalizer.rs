use std::path::Path;

use candle_core::{Device, Tensor};

use crate::config::DataConfig;
use crate::error::{Result, WeatherGraphError};
use crate::features::FeatureBundle;
use crate::geometry::ERA5_NODE_COUNT;

#[derive(Debug, Clone, PartialEq)]
pub struct Normalizer {
    pub temporal: FeatureBundle,
    pub surface: FeatureBundle,
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
    pub orography: Vec<f32>,
    pub landsea: Vec<f32>,
}

impl Normalizer {
    pub fn load(data_dir: impl AsRef<Path>, data: &DataConfig) -> Result<Self> {
        let data_dir = data_dir.as_ref();
        let temporal = FeatureBundle::load(data_dir.join(&data.normalizer_file))?;
        let surface = FeatureBundle::load(data_dir.join(&data.orography_landsea_file))?;
        let means = temporal
            .get_f32(&["means"])?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let stds = temporal
            .get_f32(&["stds"])?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let orography = surface
            .get_f32(&["orography"])?
            .into_shape_with_order(ERA5_NODE_COUNT)
            .map_err(|error| WeatherGraphError::InvalidConfig(error.to_string()))?
            .to_vec();
        let landsea = surface
            .get_f32(&["landsea"])?
            .into_shape_with_order(ERA5_NODE_COUNT)
            .map_err(|error| WeatherGraphError::InvalidConfig(error.to_string()))?
            .to_vec();

        Ok(Self {
            temporal,
            surface,
            means,
            stds,
            orography,
            landsea,
        })
    }

    pub fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.dims2()?.1 != self.means.len() {
            return Ok(tensor.clone());
        }

        let mut values = tensor.to_vec2::<f32>()?;
        for row in &mut values {
            for (index, value) in row.iter_mut().enumerate() {
                *value = (*value - self.means[index]) / self.stds[index];
            }
        }
        let width = self.means.len();
        let flattened = values.into_iter().flatten().collect::<Vec<_>>();
        let batch = flattened.len() / width;
        Ok(Tensor::from_vec(flattened, (batch, width), &Device::Cpu)?)
    }

    pub fn denormalize(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.dims2()?.1 != self.stds.len() {
            return Ok(tensor.clone());
        }

        let mut values = tensor.to_vec2::<f32>()?;
        for row in &mut values {
            for (index, value) in row.iter_mut().enumerate() {
                *value = (*value * self.stds[index]) + self.means[index];
            }
        }
        let width = self.stds.len();
        let flattened = values.into_iter().flatten().collect::<Vec<_>>();
        let batch = flattened.len() / width;
        Ok(Tensor::from_vec(flattened, (batch, width), &Device::Cpu)?)
    }

    pub fn device_default() -> Device {
        Device::Cpu
    }

    pub fn encoder_surface_features(&self, n_total_nodes: usize) -> (Vec<f32>, Vec<f32>) {
        let mut orography = vec![0.0_f32; n_total_nodes];
        let mut landsea = vec![0.0_f32; n_total_nodes];
        orography[..self.orography.len()].copy_from_slice(&self.orography);
        landsea[..self.landsea.len()].copy_from_slice(&self.landsea);
        (orography, landsea)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::Normalizer;
    use crate::features::FeatureBundle;

    #[test]
    fn normalizer_round_trip_matches_input() {
        let normalizer = Normalizer {
            temporal: FeatureBundle {
                source: "temporal".into(),
                arrays: Vec::new(),
            },
            surface: FeatureBundle {
                source: "surface".into(),
                arrays: Vec::new(),
            },
            means: vec![1.0, 2.0],
            stds: vec![2.0, 4.0],
            orography: vec![0.0; 4],
            landsea: vec![1.0; 4],
        };
        let input = Tensor::from_vec(vec![3.0_f32, 10.0], (1, 2), &Device::Cpu).expect("input");
        let normalized = normalizer.normalize(&input).expect("normalize");
        let restored = normalizer.denormalize(&normalized).expect("denormalize");
        assert_eq!(
            restored.to_vec2::<f32>().expect("restored"),
            vec![vec![3.0, 10.0]]
        );
    }
}
