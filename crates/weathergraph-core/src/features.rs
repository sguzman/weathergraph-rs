use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use ndarray::{ArrayD, IxDyn, OwnedRepr};
use ndarray_npy::NpzReader;

use crate::error::{Result, WeatherGraphError};

#[derive(Debug, Clone, PartialEq)]
pub enum NumericArray {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I64(ArrayD<i64>),
    I32(ArrayD<i32>),
    U32(ArrayD<u32>),
    U8(ArrayD<u8>),
}

impl NumericArray {
    pub fn dtype(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::I64(_) => "i64",
            Self::I32(_) => "i32",
            Self::U32(_) => "u32",
            Self::U8(_) => "u8",
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Self::F32(array) => array.shape().to_vec(),
            Self::F64(array) => array.shape().to_vec(),
            Self::I64(array) => array.shape().to_vec(),
            Self::I32(array) => array.shape().to_vec(),
            Self::U32(array) => array.shape().to_vec(),
            Self::U8(array) => array.shape().to_vec(),
        }
    }

    pub fn leading_dim(&self) -> Option<usize> {
        self.shape().first().copied()
    }

    pub fn to_u32_vec(&self, name: &str) -> Result<Vec<u32>> {
        match self {
            Self::U32(array) => Ok(array.iter().copied().collect()),
            Self::I32(array) => array
                .iter()
                .map(|value| {
                    u32::try_from(*value).map_err(|_| {
                        WeatherGraphError::InvalidConfig(format!(
                            "negative value in `{name}` cannot be converted to u32"
                        ))
                    })
                })
                .collect(),
            Self::I64(array) => array
                .iter()
                .map(|value| {
                    u32::try_from(*value).map_err(|_| {
                        WeatherGraphError::InvalidConfig(format!(
                            "value in `{name}` cannot be converted to u32"
                        ))
                    })
                })
                .collect(),
            other => Err(WeatherGraphError::UnsupportedDtype {
                name: name.to_owned(),
                dtype: other.dtype().to_owned(),
            }),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn as_f32(&self, name: &str) -> Result<ArrayD<f32>> {
        match self {
            Self::F32(array) => Ok(array.clone()),
            Self::F64(array) => Ok(array.mapv(|value| value as f32)),
            other => Err(WeatherGraphError::UnsupportedDtype {
                name: name.to_owned(),
                dtype: other.dtype().to_owned(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NamedArray {
    pub name: String,
    pub data: NumericArray,
}

impl NamedArray {
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape()
    }

    #[must_use]
    pub fn stem(&self) -> &str {
        self.name.trim_end_matches(".npy")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArraySummary {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactSummary {
    pub file_name: String,
    pub path: PathBuf,
    pub arrays: Vec<ArraySummary>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureBundle {
    pub source: PathBuf,
    pub arrays: Vec<NamedArray>,
}

impl FeatureBundle {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let arrays = load_npz_gz(&path)?;
        Ok(Self {
            source: path,
            arrays,
        })
    }

    pub fn summary(&self) -> ArtifactSummary {
        ArtifactSummary {
            file_name: self
                .source
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
            path: self.source.clone(),
            arrays: self
                .arrays
                .iter()
                .map(|entry| ArraySummary {
                    name: entry.stem().to_owned(),
                    dtype: entry.data.dtype().to_owned(),
                    shape: entry.shape(),
                })
                .collect(),
        }
    }

    pub fn first_array(&self) -> Option<&NamedArray> {
        self.arrays.first()
    }

    pub fn get(&self, candidate: &str) -> Result<&NamedArray> {
        self.arrays
            .iter()
            .find(|entry| entry.stem().eq_ignore_ascii_case(candidate))
            .ok_or_else(|| WeatherGraphError::MissingArtifact {
                name: candidate.to_owned(),
                path: self.source.clone(),
            })
    }

    pub fn get_any<'a>(&'a self, candidates: &[&str]) -> Result<&'a NamedArray> {
        for candidate in candidates {
            if let Ok(entry) = self.get(candidate) {
                return Ok(entry);
            }
        }

        Err(WeatherGraphError::MissingArtifact {
            name: candidates.join("/"),
            path: self.source.clone(),
        })
    }

    pub fn get_f32(&self, candidates: &[&str]) -> Result<ArrayD<f32>> {
        self.get_any(candidates)?.data.as_f32(&candidates.join("/"))
    }
}

pub fn load_npz_gz(path: impl AsRef<Path>) -> Result<Vec<NamedArray>> {
    let file = File::open(path.as_ref())?;
    let decoder = GzDecoder::new(file);
    let mut decompressed = Vec::new();
    let mut decoder = decoder;
    decoder.read_to_end(&mut decompressed)?;
    let cursor = Cursor::new(decompressed);
    let mut reader = NpzReader::new(cursor)?;
    let names = reader.names()?;
    let mut entries = Vec::with_capacity(names.len());

    for name in names {
        if let Ok(array) = reader.by_name::<OwnedRepr<f32>, IxDyn>(&name) {
            entries.push(NamedArray {
                name,
                data: NumericArray::F32(array),
            });
            continue;
        }

        if let Ok(array) = reader.by_name::<OwnedRepr<f64>, IxDyn>(&name) {
            entries.push(NamedArray {
                name,
                data: NumericArray::F64(array),
            });
            continue;
        }

        if let Ok(array) = reader.by_name::<OwnedRepr<i64>, IxDyn>(&name) {
            entries.push(NamedArray {
                name,
                data: NumericArray::I64(array),
            });
            continue;
        }

        if let Ok(array) = reader.by_name::<OwnedRepr<i32>, IxDyn>(&name) {
            entries.push(NamedArray {
                name,
                data: NumericArray::I32(array),
            });
            continue;
        }

        if let Ok(array) = reader.by_name::<OwnedRepr<u32>, IxDyn>(&name) {
            entries.push(NamedArray {
                name,
                data: NumericArray::U32(array),
            });
            continue;
        }

        if let Ok(array) = reader.by_name::<OwnedRepr<u8>, IxDyn>(&name) {
            entries.push(NamedArray {
                name,
                data: NumericArray::U8(array),
            });
            continue;
        }

        return Err(WeatherGraphError::UnsupportedDtype {
            name,
            dtype: "unknown".to_owned(),
        });
    }

    Ok(entries)
}

pub fn summarize_data_dir(data_dir: impl AsRef<Path>) -> Result<Vec<ArtifactSummary>> {
    let mut paths = fs::read_dir(data_dir.as_ref())?
        .map(|entry| entry.map(|value| value.path()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    paths.retain(|path| {
        path.file_name()
            .is_some_and(|name| name.to_string_lossy().ends_with(".npz.gz"))
    });
    paths.sort();

    paths
        .iter()
        .map(|path| FeatureBundle::load(path).map(|bundle| bundle.summary()))
        .collect()
}

pub fn find_artifact_path(data_dir: impl AsRef<Path>, candidates: &[&str]) -> Result<PathBuf> {
    let data_dir = data_dir.as_ref();

    for candidate in candidates {
        let path = data_dir.join(candidate);
        if path.exists() {
            return Ok(path);
        }
    }

    let available = fs::read_dir(data_dir)?
        .map(|entry| entry.map(|value| value.file_name().to_string_lossy().into_owned()))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Err(WeatherGraphError::MissingArtifact {
        name: format!("one of {}", candidates.join(", ")),
        path: PathBuf::from(available.join(", ")),
    })
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{Cursor, Write};

    use flate2::{Compression, write::GzEncoder};
    use ndarray::{Array1, Array2};
    use ndarray_npy::NpzWriter;
    use tempfile::tempdir;

    use super::{NumericArray, load_npz_gz};

    #[test]
    fn loads_npz_gz_shapes() {
        let temp_dir = tempdir().expect("tempdir");
        let path = temp_dir.path().join("fixture.npz.gz");
        let mut writer = NpzWriter::new(Cursor::new(Vec::<u8>::new()));
        writer
            .add_array("values", &Array2::<f32>::zeros((2, 3)))
            .expect("write values");
        writer
            .add_array("indices", &Array1::<i64>::from_vec(vec![1, 2, 3]))
            .expect("write indices");
        let cursor = writer.finish().expect("finish npz");
        let file = File::create(&path).expect("create fixture");
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder
            .write_all(&cursor.into_inner())
            .expect("compress fixture");
        encoder.finish().expect("finish gzip");

        let entries = load_npz_gz(&path).expect("load fixture");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].shape(), vec![2, 3]);
        assert_eq!(entries[1].shape(), vec![3]);
        assert!(matches!(entries[1].data, NumericArray::I64(_)));
    }
}
