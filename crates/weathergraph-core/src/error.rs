use std::path::PathBuf;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, WeatherGraphError>;

#[derive(Debug, Error)]
pub enum WeatherGraphError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("npz read error: {0}")]
    ReadNpz(#[from] ndarray_npy::ReadNpzError),
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),
    #[error("netcdf error: {0}")]
    Netcdf(#[from] netcdf::Error),
    #[error("safe tensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("time parse error: {0}")]
    TimeParse(#[from] chrono::ParseError),
    #[error("toml parse error: {0}")]
    TomlParse(#[from] toml::de::Error),
    #[error("missing artifact `{name}` in `{path}`")]
    MissingArtifact { name: String, path: PathBuf },
    #[error("unsupported artifact dtype `{dtype}` for `{name}`")]
    UnsupportedDtype { name: String, dtype: String },
    #[error("shape mismatch for `{name}`: expected {expected}, got {actual}")]
    ShapeMismatch {
        name: String,
        expected: String,
        actual: String,
    },
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("not yet implemented: {0}")]
    NotYetImplemented(String),
}
