use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Result, WeatherGraphError};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Config {
    pub artifacts: ArtifactPaths,
    pub data: DataConfig,
    pub model: ModelConfig,
}

impl Config {
    pub fn from_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            artifacts: ArtifactPaths::new(data_dir.into()),
            data: DataConfig::default(),
            model: ModelConfig::default(),
        }
    }

    pub fn artifact_file(&self, name: &str) -> PathBuf {
        self.artifacts.data_dir.join(name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataConfig {
    pub weights_file: String,
    pub normalizer_file: String,
    pub senders_receivers_encoder: String,
    pub senders_receivers_processor: String,
    pub senders_receivers_decoder: String,
    pub node_features_e: String,
    pub edge_features_e: String,
    pub node_features_p: String,
    pub edge_features_p: String,
    pub node_features_d: String,
    pub edge_features_d: String,
    pub orography_landsea_file: String,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            weights_file: "good_era5_forecast_batch001_feats0256_blocks009_steps12_stride02_noise0.0000_l10.0000000_lr0.000003_lrd0_ethereal-brook-316_val1.0347996_idx030200.pkl".to_owned(),
            normalizer_file: "temporal_normalizer_rk-era5-data_zarr-era5_1979begin_2020end_03hr_6phys_181lat_360lon_13levels_blosc1comp_Corder_monolith.npz.gz".to_owned(),
            senders_receivers_encoder: "senders_receivers_encoder.npz.gz".to_owned(),
            senders_receivers_processor: "senders_receivers_processor.npz.gz".to_owned(),
            senders_receivers_decoder: "senders_receivers_decoder.npz.gz".to_owned(),
            node_features_e: "node_features_n71042_e112246_s-8416688801745003395_r-6736346125390000850.npz.gz".to_owned(),
            edge_features_e: "edge_features_n71042_e112246_s-8416688801745003395_r-6736346125390000850.npz.gz".to_owned(),
            node_features_p: "node_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz".to_owned(),
            edge_features_p: "edge_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz".to_owned(),
            node_features_d: "node_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz".to_owned(),
            edge_features_d: "edge_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz".to_owned(),
            orography_landsea_file: "orography_landsea.npz.gz".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactPaths {
    pub data_dir: PathBuf,
    pub weights_file: Option<PathBuf>,
}

impl ArtifactPaths {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            weights_file: None,
        }
    }

    #[must_use]
    pub fn with_weights(mut self, weights_file: impl Into<PathBuf>) -> Self {
        self.weights_file = Some(weights_file.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelConfig {
    pub hidden_dim: usize,
    pub processor_blocks: usize,
    pub use_layer_norm: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            processor_blocks: 9,
            use_layer_norm: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForecastInputKind {
    Era5,
    Opendata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForecastRequest {
    pub init: DateTime<Utc>,
    pub steps: usize,
    pub input: ForecastInputKind,
    pub out: PathBuf,
}

impl ForecastRequest {
    pub fn validate(&self) -> Result<()> {
        if self.steps == 0 {
            return Err(WeatherGraphError::InvalidConfig(
                "forecast steps must be greater than zero".to_owned(),
            ));
        }

        let Some(parent) = self.out.parent() else {
            return Err(WeatherGraphError::InvalidConfig(format!(
                "forecast output path `{}` must have a parent directory",
                self.out.display()
            )));
        };

        if !parent.as_os_str().is_empty() && !Path::new(parent).exists() {
            return Err(WeatherGraphError::InvalidConfig(format!(
                "forecast output directory `{}` does not exist",
                parent.display()
            )));
        }

        Ok(())
    }
}
