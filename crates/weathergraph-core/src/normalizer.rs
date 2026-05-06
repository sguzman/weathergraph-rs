use std::path::Path;

use candle_core::{Device, Tensor};

use crate::error::Result;
use crate::features::{FeatureBundle, find_artifact_path};

#[derive(Debug, Clone, PartialEq)]
pub struct Normalizer {
    pub temporal: FeatureBundle,
    pub surface: FeatureBundle,
}

impl Normalizer {
    pub fn load(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref();
        let temporal_path = find_artifact_path(
            data_dir,
            &[
                "temporal_normalizer.npz.gz",
                "temporal_normalizer_rk-era5-data_zarr-era5_1979begin_2020end_03hr_6phys_181lat_360lon_13levels_blosc1comp_Corder_monolith.npz.gz",
            ],
        )?;
        let surface_path = find_artifact_path(data_dir, &["orography_landsea.npz.gz"])?;

        Ok(Self {
            temporal: FeatureBundle::load(temporal_path)?,
            surface: FeatureBundle::load(surface_path)?,
        })
    }

    pub fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    pub fn denormalize(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    pub fn device_default() -> Device {
        Device::Cpu
    }
}
