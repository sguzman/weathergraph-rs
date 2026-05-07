use candle_core::{Device, Tensor};
use chrono::Duration;

use crate::config::{Config, ForecastInputKind, ForecastRequest};
use crate::error::{Result, WeatherGraphError};
use crate::geometry::{ERA5_LAT_COUNT, ERA5_LON_COUNT};
use crate::graph::GraphSet;
use crate::model::KeislerGnn;
use crate::normalizer::Normalizer;
use crate::solar::{SOLAR_TIME_SHIFTS_HOURS, centered_solar_features, day_of_year_feature};

const VARNAMES: [&str; 6] = [
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "geopotential",
];
const LEVELS: [i32; 13] = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
];

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

        if request
            .out
            .extension()
            .is_none_or(|extension| extension != "nc")
        {
            return Err(WeatherGraphError::InvalidConfig(
                "forecast output must use a .nc extension".to_owned(),
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
        let expected_channels = self.config.model.input_channels;
        if feature_dim != expected_channels {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "runner state".to_owned(),
                expected: expected_channels.to_string(),
                actual: feature_dim.to_string(),
            });
        }
        if n_nodes != self.graphs.n_total_nodes {
            return Err(WeatherGraphError::ShapeMismatch {
                name: "runner state nodes".to_owned(),
                expected: self.graphs.n_total_nodes.to_string(),
                actual: n_nodes.to_string(),
            });
        }

        let step_index = i64::try_from(step_index).unwrap_or(i64::MAX / 6);
        let valid_time = request.init + Duration::hours(step_index.saturating_mul(6));
        let solar = self.centered_solar_tensor(valid_time)?;
        let doy = self.day_of_year_tensor(valid_time)?;
        let (orography_values, landsea_values) = self
            .normalizer
            .encoder_surface_features(self.graphs.n_total_nodes);
        let orography = Tensor::from_vec(
            orography_values,
            (self.graphs.n_total_nodes, 1),
            &self.device,
        )?;
        let landsea =
            Tensor::from_vec(landsea_values, (self.graphs.n_total_nodes, 1), &self.device)?;
        let normalized = self.normalizer.normalize(state)?;
        let output = self.model.one_step_graph(
            &normalized,
            &self.graphs,
            &solar,
            &doy,
            &orography,
            &landsea,
        )?;
        self.normalizer.denormalize(&output)
    }

    pub fn run_forecast(&self, request: &ForecastRequest) -> Result<()> {
        self.validate_request(request)?;
        let mut current_state = self.load_initial_state(request)?;
        let mut states = Vec::with_capacity(request.steps + 1);
        states.push(self.era5_state_rows(&current_state)?);

        for step_index in 0..request.steps {
            current_state = self.one_step(&current_state, step_index, request)?;
            states.push(self.era5_state_rows(&current_state)?);
        }

        self.write_forecast_netcdf(request, &states)
    }

    fn centered_solar_tensor(&self, valid_time: chrono::DateTime<chrono::Utc>) -> Result<Tensor> {
        let coslat = tensor_first_column(
            &self.graphs.encoder.node_tensors.coslat,
            self.graphs.n_era5_nodes,
        )?;
        let sinlat = tensor_first_column(
            &self.graphs.encoder.node_tensors.sinlat,
            self.graphs.n_era5_nodes,
        )?;
        let coslon = tensor_first_column(
            &self.graphs.encoder.node_tensors.coslon,
            self.graphs.n_era5_nodes,
        )?;
        let sinlon = tensor_first_column(
            &self.graphs.encoder.node_tensors.sinlon,
            self.graphs.n_era5_nodes,
        )?;
        let mut solar_values =
            centered_solar_features(&coslat, &sinlat, &coslon, &sinlon, valid_time);
        solar_values.extend(vec![
            0.0_f32;
            self.graphs.n_h3_nodes * SOLAR_TIME_SHIFTS_HOURS.len()
        ]);
        Tensor::from_vec(
            solar_values,
            (self.graphs.n_total_nodes, SOLAR_TIME_SHIFTS_HOURS.len()),
            &self.device,
        )
        .map_err(Into::into)
    }

    fn day_of_year_tensor(&self, valid_time: chrono::DateTime<chrono::Utc>) -> Result<Tensor> {
        let mut values = vec![0.0_f32; self.graphs.n_total_nodes];
        values[..self.graphs.n_era5_nodes].fill(day_of_year_feature(valid_time));
        Tensor::from_vec(values, (self.graphs.n_total_nodes, 1), &self.device).map_err(Into::into)
    }

    fn load_initial_state(&self, request: &ForecastRequest) -> Result<Tensor> {
        match request.input {
            ForecastInputKind::Era5 => self.load_era5_input_state(),
            ForecastInputKind::Opendata => Err(WeatherGraphError::NotYetImplemented(
                "opendata input is not yet supported in the Rust runner; use a prepared local ERA5-style NetCDF file".to_owned(),
            )),
        }
    }

    fn load_era5_input_state(&self) -> Result<Tensor> {
        let path = self
            .config
            .artifacts
            .data_dir
            .join(&self.config.data.era5_input_file);
        let file = netcdf::open(path)?;
        let mut channels = Vec::with_capacity(VARNAMES.len() * LEVELS.len());

        for variable_name in VARNAMES {
            let variable =
                file.variable(variable_name)
                    .ok_or_else(|| WeatherGraphError::MissingArtifact {
                        name: variable_name.to_owned(),
                        path: self
                            .config
                            .artifacts
                            .data_dir
                            .join(&self.config.data.era5_input_file),
                    })?;
            let dims = variable.dimensions();
            if dims.len() != 4
                || dims[0].len() != 1
                || dims[1].len() != LEVELS.len()
                || dims[2].len() != ERA5_LAT_COUNT
                || dims[3].len() != ERA5_LON_COUNT
            {
                return Err(WeatherGraphError::ShapeMismatch {
                    name: variable_name.to_owned(),
                    expected: format!(
                        "[1, {}, {}, {}]",
                        LEVELS.len(),
                        ERA5_LAT_COUNT,
                        ERA5_LON_COUNT
                    ),
                    actual: format!(
                        "[{}, {}, {}, {}]",
                        dims[0].len(),
                        dims[1].len(),
                        dims[2].len(),
                        dims[3].len()
                    ),
                });
            }

            let values = variable.get_values::<f32, _>(..)?;
            for level_index in 0..LEVELS.len() {
                let start = level_index * ERA5_LAT_COUNT * ERA5_LON_COUNT;
                let end = start + (ERA5_LAT_COUNT * ERA5_LON_COUNT);
                channels.push(values[start..end].to_vec());
            }
        }

        let mut padded =
            vec![0.0_f32; self.graphs.n_total_nodes * self.config.model.input_channels];
        let width = self.config.model.input_channels;
        for row_index in 0..self.graphs.n_era5_nodes {
            let target_offset = row_index * width;
            for channel_index in 0..width {
                let value = channels
                    .get(channel_index)
                    .and_then(|channel| channel.get(row_index))
                    .copied()
                    .unwrap_or(0.0);
                padded[target_offset + channel_index] = value;
            }
        }
        Tensor::from_vec(
            padded,
            (self.graphs.n_total_nodes, self.config.model.input_channels),
            &self.device,
        )
        .map_err(Into::into)
    }

    fn era5_state_rows(&self, state: &Tensor) -> Result<Vec<f32>> {
        let rows = state.to_vec2::<f32>()?;
        Ok(rows
            .into_iter()
            .take(self.graphs.n_era5_nodes)
            .flatten()
            .collect::<Vec<_>>())
    }

    fn write_forecast_netcdf(&self, request: &ForecastRequest, states: &[Vec<f32>]) -> Result<()> {
        let mut file = netcdf::create(&request.out)?;
        let _time_dim = file.add_dimension("time", states.len())?;
        let _level_dim = file.add_dimension("level", LEVELS.len())?;
        let _lat_dim = file.add_dimension("latitude", ERA5_LAT_COUNT)?;
        let _lon_dim = file.add_dimension("longitude", ERA5_LON_COUNT)?;

        let mut time_variable = file.add_variable::<i64>("time", &["time"])?;
        let time_values = (0..states.len())
            .map(|index| i64::try_from(index).unwrap_or(i64::MAX).saturating_mul(6))
            .collect::<Vec<_>>();
        time_variable.put_values(&time_values, ..)?;
        time_variable.put_attribute("units", "hours since init")?;

        let mut level_variable = file.add_variable::<i32>("level", &["level"])?;
        level_variable.put_values(&LEVELS, ..)?;

        let mut latitude_variable = file.add_variable::<f32>("latitude", &["latitude"])?;
        let latitude_values = (0..ERA5_LAT_COUNT)
            .map(|index| 90.0_f32 - f32::from(u16::try_from(index).expect("latitude index")))
            .collect::<Vec<_>>();
        latitude_variable.put_values(&latitude_values, ..)?;

        let mut longitude_variable = file.add_variable::<f32>("longitude", &["longitude"])?;
        let longitude_values = (0..ERA5_LON_COUNT)
            .map(|index| f32::from(u16::try_from(index).expect("longitude index")))
            .collect::<Vec<_>>();
        longitude_variable.put_values(&longitude_values, ..)?;

        for (variable_index, variable_name) in VARNAMES.iter().enumerate() {
            let mut variable = file
                .add_variable::<f32>(variable_name, &["time", "level", "latitude", "longitude"])?;
            let mut values =
                Vec::with_capacity(states.len() * LEVELS.len() * ERA5_LAT_COUNT * ERA5_LON_COUNT);
            for state in states {
                for level_index in 0..LEVELS.len() {
                    for lat_index in 0..ERA5_LAT_COUNT {
                        for lon_index in 0..ERA5_LON_COUNT {
                            let channel_index = (variable_index * LEVELS.len()) + level_index;
                            let node_index = (lat_index * ERA5_LON_COUNT) + lon_index;
                            let state_index =
                                (node_index * self.config.model.output_channels) + channel_index;
                            values.push(state.get(state_index).copied().unwrap_or(0.0));
                        }
                    }
                }
            }
            variable.put_values(&values, ..)?;
        }

        file.add_attribute("init", request.init.to_rfc3339())?;
        file.add_attribute("input_kind", format!("{:?}", request.input))?;
        Ok(())
    }
}

fn tensor_first_column(tensor: &Tensor, rows: usize) -> Result<Vec<f32>> {
    let values = tensor.to_vec2::<f32>()?;
    Ok(values
        .into_iter()
        .take(rows)
        .map(|row| row.first().copied().unwrap_or(0.0))
        .collect::<Vec<_>>())
}
