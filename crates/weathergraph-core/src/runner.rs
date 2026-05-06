use candle_core::{Device, Tensor};

use crate::config::{Config, ForecastRequest};
use crate::error::{Result, WeatherGraphError};
use crate::graph::GraphSet;
use crate::model::KeislerGnn;
use crate::normalizer::Normalizer;
use crate::solar::solar_features;
use crate::tensor::shape_string;

#[derive(Debug, Clone)]
pub struct Runner {
    pub config: Config,
    pub graphs: GraphSet,
    pub normalizer: Normalizer,
    pub model: KeislerGnn,
    pub device: Device,
}

impl Runner {
    pub fn load(config: Config) -> Result<Self> {
        let device = Device::Cpu;
        let graphs = GraphSet::load(&config.artifacts.data_dir, &config.data)?;
        let normalizer = Normalizer::load(&config.artifacts.data_dir, &config.data)?;
        let model = if let Some(weights_file) = &config.artifacts.weights_file {
            KeislerGnn::from_safetensors(weights_file, &config.model, &device)?
        } else {
            KeislerGnn::placeholder(&config.model, &device)?
        };

        Ok(Self {
            config,
            graphs,
            normalizer,
            model,
            device,
        })
    }

    pub fn validate_request(&self, request: &ForecastRequest) -> Result<()> {
        request.validate()?;

        if self.config.artifacts.weights_file.is_none() {
            return Err(WeatherGraphError::InvalidConfig(
                "forecast requires --weights with an exported safetensors file".to_owned(),
            ));
        }

        Ok(())
    }

    pub fn one_step(
        &self,
        state: &Tensor,
        step_index: usize,
        request: &ForecastRequest,
    ) -> Result<Tensor> {
        let (n_nodes, feature_dim) = state.dims2()?;
        let expected_channels = self
            .normalizer
            .means
            .len()
            .max(self.config.model.input_channels);
        if feature_dim != expected_channels {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "runner state".to_owned(),
                expected: expected_channels.to_string(),
                actual: feature_dim.to_string(),
            });
        }
        if n_nodes == 0 {
            return Err(WeatherGraphError::InvalidConfig(
                "runner state must contain at least one node".to_owned(),
            ));
        }

        let _solar = solar_features(request.init, step_index);
        let normalized = self.normalizer.normalize(state)?;
        let output = self.model.one_step(&normalized)?;
        self.normalizer.denormalize(&output)
    }

    pub fn run_forecast(&self, request: &ForecastRequest) -> Result<()> {
        self.validate_request(request)?;
        let input_channels = self
            .normalizer
            .means
            .len()
            .max(self.config.model.input_channels);
        let _surface = self
            .normalizer
            .encoder_surface_features(self.graphs.n_total_nodes);
        let initial_state = Tensor::zeros(
            (self.graphs.n_total_nodes, input_channels),
            candle_core::DType::F32,
            &self.device,
        )?;
        let _ = self.one_step(&initial_state, 0, request)?;
        Err(WeatherGraphError::NotYetImplemented(format!(
            "autoregressive rollout and NetCDF output are deferred; one-step scaffold validated for state shape {}",
            shape_string(&[self.graphs.n_total_nodes, input_channels])
        )))
    }
}
