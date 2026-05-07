import datetime as dt
import unittest

from tools.fetch_arco_era5 import hour_index, parse_init_timestamp


class FetchArcoEra5Tests(unittest.TestCase):
    def test_parse_init_timestamp_normalizes_to_utc(self) -> None:
        parsed = parse_init_timestamp("2020-01-01T00:00:00Z")
        self.assertEqual(parsed.tzinfo, dt.timezone.utc)
        self.assertEqual(parsed.hour, 0)

    def test_parse_init_timestamp_rejects_non_hourly_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_init_timestamp("2020-01-01T00:30:00Z")

    def test_hour_index_matches_arco_epoch_hours(self) -> None:
        parsed = parse_init_timestamp("1900-01-02T03:00:00Z")
        self.assertEqual(hour_index(parsed), 27)


if __name__ == "__main__":
    unittest.main()
