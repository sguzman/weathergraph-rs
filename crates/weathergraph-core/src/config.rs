use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Result, WeatherGraphError};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Config {
    pub artifacts: ArtifactPaths,
    pub model: ModelConfig,
}

impl Config {
    pub fn from_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            artifacts: ArtifactPaths::new(data_dir.into()),
            model: ModelConfig::default(),
        }
    }

    pub fn artifact_file(&self, name: &str) -> PathBuf {
        self.artifacts.data_dir.join(name)
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
