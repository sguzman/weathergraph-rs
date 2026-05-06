use chrono::{DateTime, Datelike, Duration, Timelike, Utc};

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn solar_features(init: DateTime<Utc>, step_index: usize) -> [f32; 4] {
    let step_index = i64::try_from(step_index).unwrap_or(i64::MAX / 6);
    let valid_time = init + Duration::hours(step_index.saturating_mul(6));
    let day_fraction = (valid_time.hour() as f32
        + valid_time.minute() as f32 / 60.0
        + valid_time.second() as f32 / 3600.0)
        / 24.0;
    let seasonal_fraction = valid_time.ordinal0() as f32 / 365.0;
    [
        (std::f32::consts::TAU * day_fraction).sin(),
        (std::f32::consts::TAU * day_fraction).cos(),
        (std::f32::consts::TAU * seasonal_fraction).sin(),
        (std::f32::consts::TAU * seasonal_fraction).cos(),
    ]
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::solar_features;

    #[test]
    fn solar_features_are_stable() {
        let init = Utc
            .with_ymd_and_hms(2020, 1, 1, 0, 0, 0)
            .single()
            .expect("time");
        let features = solar_features(init, 1);
        assert_eq!(features.len(), 4);
        assert!((features[1] - 0.0).abs() <= 1.0);
    }
}
