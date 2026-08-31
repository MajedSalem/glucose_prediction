import pandas as pd

from glucose_prediction.data.ohio_t1dm import _distribute_bolus


def _bolus_df(rows):
    return pd.DataFrame(rows, columns=["dose", "ts_begin", "ts_end"])


def test_distribute_bolus_instantaneous_goes_to_nearest_bucket():
    grid = pd.date_range("2024-01-01 00:00", periods=5, freq="5min")
    ts = pd.Timestamp("2024-01-01 00:10")
    bolus_df = _bolus_df([{"dose": 4.0, "ts_begin": ts, "ts_end": ts}])
    result = _distribute_bolus(grid, bolus_df)
    assert result.tolist() == [0.0, 0.0, 4.0, 0.0, 0.0]


def test_distribute_bolus_spreads_proportionally_across_overlapping_buckets():
    grid = pd.date_range("2024-01-01 00:00", periods=4, freq="5min")  # buckets [0,5) [5,10) [10,15) [15,20)
    ts_begin = pd.Timestamp("2024-01-01 00:05")
    ts_end = pd.Timestamp("2024-01-01 00:15")  # 10-min duration, exactly buckets 1 and 2
    bolus_df = _bolus_df([{"dose": 10.0, "ts_begin": ts_begin, "ts_end": ts_end}])
    result = _distribute_bolus(grid, bolus_df)
    assert result.tolist() == [0.0, 5.0, 5.0, 0.0]


def test_distribute_bolus_handles_partial_bucket_overlap():
    grid = pd.date_range("2024-01-01 00:00", periods=3, freq="5min")  # buckets [0,5) [5,10) [10,15)
    ts_begin = pd.Timestamp("2024-01-01 00:02:30")
    ts_end = pd.Timestamp("2024-01-01 00:07:30")  # 5-min duration, half in bucket 0, half in bucket 1
    bolus_df = _bolus_df([{"dose": 8.0, "ts_begin": ts_begin, "ts_end": ts_end}])
    result = _distribute_bolus(grid, bolus_df)
    assert result.tolist() == [4.0, 4.0, 0.0]


def test_distribute_bolus_sums_overlapping_events():
    grid = pd.date_range("2024-01-01 00:00", periods=3, freq="5min")
    ts = pd.Timestamp("2024-01-01 00:00")
    bolus_df = _bolus_df([
        {"dose": 2.0, "ts_begin": ts, "ts_end": ts},
        {"dose": 3.0, "ts_begin": ts, "ts_end": ts},
    ])
    result = _distribute_bolus(grid, bolus_df)
    assert result.tolist() == [5.0, 0.0, 0.0]


def test_distribute_bolus_empty_returns_zeros():
    grid = pd.date_range("2024-01-01 00:00", periods=3, freq="5min")
    result = _distribute_bolus(grid, _bolus_df([]))
    assert result.tolist() == [0.0, 0.0, 0.0]
