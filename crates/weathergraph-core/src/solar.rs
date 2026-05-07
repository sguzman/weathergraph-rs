use chrono::{DateTime, Datelike, Duration, Timelike, Utc};

pub const SOLAR_TIME_SHIFTS_HOURS: [i64; 11] = [-12, -9, -6, -3, -1, 0, 1, 3, 6, 9, 12];

#[allow(clippy::cast_precision_loss)]
pub fn centered_solar_features(
    coslat: &[f32],
    sinlat: &[f32],
    coslon: &[f32],
    sinlon: &[f32],
    valid_time: DateTime<Utc>,
) -> Vec<f32> {
    let n_nodes = coslat.len();
    let mut values = Vec::with_capacity(n_nodes * SOLAR_TIME_SHIFTS_HOURS.len());

    for node_index in 0..n_nodes {
        let lat_deg = sinlat[node_index].atan2(coslat[node_index]).to_degrees();
        let lon_deg = sinlon[node_index].atan2(coslon[node_index]).to_degrees();
        for shift_hours in SOLAR_TIME_SHIFTS_HOURS {
            let shifted = valid_time + Duration::hours(shift_hours);
            values.push(toa_solar_rad_approx(lat_deg, lon_deg, shifted));
        }
    }

    values
}

#[allow(clippy::cast_precision_loss)]
pub fn day_of_year_feature(valid_time: DateTime<Utc>) -> f32 {
    valid_time.ordinal() as f32 / 365.0
}

#[allow(clippy::cast_precision_loss)]
fn toa_solar_rad_approx(lat_deg: f32, lon_deg: f32, when: DateTime<Utc>) -> f32 {
    let altitude = solar_altitude_approx(lat_deg, lon_deg, when).max(1.0e-3);
    q0(when.ordinal()) * altitude.to_radians().sin()
}

#[allow(clippy::cast_precision_loss)]
fn solar_altitude_approx(lat_deg: f32, lon_deg: f32, when: DateTime<Utc>) -> f32 {
    let day_of_year = when.ordinal() as f32;
    let declination_deg =
        23.45 * ((2.0 * std::f32::consts::PI / 365.0) * (day_of_year - 81.0)).sin();
    let lat_rad = lat_deg.to_radians();
    let declination_rad = declination_deg.to_radians();
    let hour_angle_deg = hour_angle(when, lon_deg);
    let a = lat_rad.cos() * declination_rad.cos() * hour_angle_deg.to_radians().cos();
    let b = lat_rad.sin() * declination_rad.sin();
    (a + b).asin().to_degrees()
}

#[allow(clippy::cast_precision_loss)]
fn q0(day_of_year: u32) -> f32 {
    1.0 + 0.034 * ((2.0 * std::f32::consts::PI / 365.24) * day_of_year as f32).cos()
}

#[allow(clippy::cast_precision_loss)]
fn hour_angle(when: DateTime<Utc>, lon_deg: f32) -> f32 {
    let minutes = (60 * when.hour() + when.minute()) as f32;
    let solar_time = (minutes + (4.0 * lon_deg) + eq_of_time(when.ordinal())) / 60.0;
    15.0 * (solar_time - 12.0)
}

#[allow(clippy::cast_precision_loss)]
fn eq_of_time(day_of_year: u32) -> f32 {
    let angle = 2.0 * std::f32::consts::PI / 364.0 * (day_of_year as f32 - 81.0);
    9.87 * (2.0 * angle).sin() - 7.53 * angle.cos() - 1.5 * angle.sin()
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::{SOLAR_TIME_SHIFTS_HOURS, centered_solar_features, day_of_year_feature};

    #[test]
    fn solar_features_match_expected_width() {
        let init = Utc
            .with_ymd_and_hms(2020, 1, 1, 0, 0, 0)
            .single()
            .expect("time");
        let values = centered_solar_features(&[1.0], &[0.0], &[1.0], &[0.0], init);
        assert_eq!(values.len(), SOLAR_TIME_SHIFTS_HOURS.len());
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn day_of_year_is_normalized() {
        let init = Utc
            .with_ymd_and_hms(2020, 6, 1, 0, 0, 0)
            .single()
            .expect("time");
        let feature = day_of_year_feature(init);
        assert!(feature > 0.0);
        assert!(feature <= 1.0);
    }
}
