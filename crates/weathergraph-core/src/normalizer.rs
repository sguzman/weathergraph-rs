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
    pub fn load(
        data_dir: impl AsRef<Path>,
        data: &DataConfig,
        input_channels: usize,
    ) -> Result<Self> {
        let data_dir = data_dir.as_ref();
        let temporal = FeatureBundle::load(data_dir.join(&data.normalizer_file))?;
        let surface = FeatureBundle::load(data_dir.join(&data.orography_landsea_file))?;
        let means = temporal
            .get_f32(&["means"])?
            .iter()
            .copied()
            .take(input_channels)
            .collect::<Vec<_>>();
        let stds = temporal
            .get_f32(&["stds"])?
            .iter()
            .copied()
            .take(input_channels)
            .collect::<Vec<_>>();

        if means.len() != input_channels || stds.len() != input_channels {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "temporal normalizer".to_owned(),
                expected: input_channels.to_string(),
                actual: format!("{}/{}", means.len(), stds.len()),
            });
        }
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
        let width = self.means.len();
        let means = Tensor::from_vec(self.means.clone(), (1, width), tensor.device())?;
        let stds = Tensor::from_vec(self.stds.clone(), (1, width), tensor.device())?;
        tensor
            .broadcast_sub(&means)?
            .broadcast_div(&stds)
            .map_err(Into::into)
    }

    pub fn denormalize(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.dims2()?.1 != self.stds.len() {
            return Ok(tensor.clone());
        }
        let width = self.stds.len();
        let means = Tensor::from_vec(self.means.clone(), (1, width), tensor.device())?;
        let stds = Tensor::from_vec(self.stds.clone(), (1, width), tensor.device())?;
        tensor
            .broadcast_mul(&stds)?
            .broadcast_add(&means)
            .map_err(Into::into)
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
    use std::fs;

    use candle_core::{Device, Tensor};
    use ndarray::Array;
    use tempfile::tempdir;

    use super::Normalizer;
    use crate::config::DataConfig;
    use crate::features::{FeatureBundle, load_npz_gz};

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

    #[test]
    fn load_trims_temporal_stats_to_requested_channel_count() {
        let dir = tempdir().expect("tempdir");
        let temporal_path = dir.path().join("temporal_normalizer.npz.gz");
        let surface_path = dir.path().join("orography_landsea.npz.gz");

        write_npz(
            &temporal_path,
            &[
                (
                    "means.npy",
                    Array::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn(),
                ),
                (
                    "stds.npy",
                    Array::from_vec(vec![4.0_f32, 5.0, 6.0]).into_dyn(),
                ),
            ],
        );
        write_npz(
            &surface_path,
            &[
                (
                    "orography.npy",
                    Array::from_elem((181, 360), 0.0_f32).into_dyn(),
                ),
                (
                    "landsea.npy",
                    Array::from_elem((181, 360), 1.0_f32).into_dyn(),
                ),
            ],
        );

        let data = DataConfig {
            normalizer_file: "temporal_normalizer.npz.gz".to_owned(),
            orography_landsea_file: "orography_landsea.npz.gz".to_owned(),
            ..DataConfig::default()
        };

        let normalizer = Normalizer::load(dir.path(), &data, 2).expect("load trimmed normalizer");
        assert_eq!(normalizer.means, vec![1.0, 2.0]);
        assert_eq!(normalizer.stds, vec![4.0, 5.0]);

        let loaded = load_npz_gz(temporal_path).expect("sanity read npz");
        assert_eq!(loaded.len(), 2);
    }

    fn write_npz(path: &std::path::Path, arrays: &[(&str, ndarray::ArrayD<f32>)]) {
        let file = fs::File::create(path).expect("create npz");
        let mut writer = ndarray_npy::NpzWriter::new(file);
        for (name, array) in arrays {
            writer.add_array(*name, array).expect("add array");
        }
        writer.finish().expect("finish npz");
    }
}
