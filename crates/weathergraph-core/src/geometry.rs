use h3o::{CellIndex, LatLng, Resolution};

use crate::error::{Result, WeatherGraphError};

pub const ERA5_LAT_COUNT: usize = 181;
pub const ERA5_LON_COUNT: usize = 360;
pub const ERA5_NODE_COUNT: usize = ERA5_LAT_COUNT * ERA5_LON_COUNT;
pub const H3_RESOLUTION: u8 = 2;
pub const H3_NODE_COUNT: usize = 5_882;
pub const TOTAL_NODE_COUNT: usize = ERA5_NODE_COUNT + H3_NODE_COUNT;

#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub xyz: Vec<[f32; 3]>,
    pub latlonr: Vec<[f32; 3]>,
}

impl Geometry {
    pub fn len(&self) -> usize {
        self.xyz.len()
    }

    pub fn is_empty(&self) -> bool {
        self.xyz.is_empty()
    }

    #[must_use]
    pub fn concat(&self, other: &Self) -> Self {
        let mut xyz = self.xyz.clone();
        xyz.extend_from_slice(&other.xyz);

        let mut latlonr = self.latlonr.clone();
        latlonr.extend_from_slice(&other.latlonr);

        Self { xyz, latlonr }
    }
}

pub fn xyz_from_latlonr(lat_deg: f32, lon_deg: f32, radius: f32) -> [f32; 3] {
    let lat_rad = lat_deg.to_radians();
    let lon_rad = lon_deg.to_radians();
    let x = radius * lat_rad.cos() * lon_rad.cos();
    let y = radius * lat_rad.cos() * lon_rad.sin();
    let z = radius * lat_rad.sin();
    [x, y, z]
}

pub fn build_era5_geometry() -> Geometry {
    let mut xyz = Vec::with_capacity(ERA5_NODE_COUNT);
    let mut latlonr = Vec::with_capacity(ERA5_NODE_COUNT);

    for lat in (-(90_i16))..=90 {
        let lat = -f32::from(lat);
        for lon in 0..ERA5_LON_COUNT {
            let lon = f32::from(u16::try_from(lon).expect("ERA5 longitude index fits into u16"));
            latlonr.push([lat, lon, 1.0]);
            xyz.push(xyz_from_latlonr(lat, lon, 1.0));
        }
    }

    Geometry { xyz, latlonr }
}

#[allow(clippy::cast_possible_truncation)]
pub fn build_h3_geometry() -> Result<Geometry> {
    let resolution = Resolution::try_from(H3_RESOLUTION).map_err(|error| {
        WeatherGraphError::InvalidConfig(format!("invalid H3 resolution {H3_RESOLUTION}: {error}"))
    })?;

    let mut cells = CellIndex::base_cells()
        .flat_map(|cell| cell.children(resolution))
        .collect::<Vec<_>>();
    cells.sort_unstable();

    if cells.len() != H3_NODE_COUNT {
        return Err(WeatherGraphError::ShapeMismatch {
            name: "h3 geometry".to_owned(),
            expected: H3_NODE_COUNT.to_string(),
            actual: cells.len().to_string(),
        });
    }

    let mut xyz = Vec::with_capacity(cells.len());
    let mut latlonr = Vec::with_capacity(cells.len());

    for cell in cells {
        let center = LatLng::from(cell);
        let lat = center.lat() as f32;
        let lon = center.lng() as f32;
        latlonr.push([lat, lon, 1.0]);
        xyz.push(xyz_from_latlonr(lat, lon, 1.0));
    }

    Ok(Geometry { xyz, latlonr })
}

pub fn verify_geometry_counts() -> Result<()> {
    let era5 = build_era5_geometry();
    let h3 = build_h3_geometry()?;
    let total = era5.len() + h3.len();

    if era5.len() != ERA5_NODE_COUNT {
        return Err(WeatherGraphError::ShapeMismatch {
            name: "era5 geometry".to_owned(),
            expected: ERA5_NODE_COUNT.to_string(),
            actual: era5.len().to_string(),
        });
    }

    if h3.len() != H3_NODE_COUNT {
        return Err(WeatherGraphError::ShapeMismatch {
            name: "h3 geometry".to_owned(),
            expected: H3_NODE_COUNT.to_string(),
            actual: h3.len().to_string(),
        });
    }

    if total != TOTAL_NODE_COUNT {
        return Err(WeatherGraphError::ShapeMismatch {
            name: "combined geometry".to_owned(),
            expected: TOTAL_NODE_COUNT.to_string(),
            actual: total.to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ERA5_NODE_COUNT, H3_NODE_COUNT, TOTAL_NODE_COUNT, build_era5_geometry, build_h3_geometry,
        verify_geometry_counts, xyz_from_latlonr,
    };

    #[test]
    fn xyz_matches_unit_sphere_axes() {
        let equator = xyz_from_latlonr(0.0, 0.0, 1.0);
        let pole = xyz_from_latlonr(90.0, 0.0, 1.0);
        assert!((equator[0] - 1.0).abs() < 1.0e-6);
        assert!(equator[1].abs() < 1.0e-6);
        assert!(equator[2].abs() < 1.0e-6);
        assert!(pole[0].abs() < 1.0e-6);
        assert!(pole[1].abs() < 1.0e-6);
        assert!((pole[2] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn era5_count_matches_expected() {
        assert_eq!(build_era5_geometry().len(), ERA5_NODE_COUNT);
    }

    #[test]
    fn h3_count_matches_expected() {
        assert_eq!(
            build_h3_geometry().expect("h3 geometry").len(),
            H3_NODE_COUNT
        );
    }

    #[test]
    fn combined_counts_match_expected() {
        verify_geometry_counts().expect("verify counts");
        let total = build_era5_geometry().len() + build_h3_geometry().expect("h3 geometry").len();
        assert_eq!(total, TOTAL_NODE_COUNT);
    }
}
