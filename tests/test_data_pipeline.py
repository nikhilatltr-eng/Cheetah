import os
import sys
import datetime
import pandas as pd
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_cleaner import DataCleaner
from data.timeframe_generator import TimeframeGenerator

def test_data_pipeline_cleaning_and_aggregation():
    """
    Verifies that the historical data pipeline correctly:
      1. Removes duplicates and sorts chronologically.
      2. Validates and repairs OHLC price consistency bounds (High >= max, Low <= min).
      3. Performs gap filling on missing periods without pricing fabrication.
      4. Aggregates M1 candles into correct counts for resampled timeframes.
    """
    # 1. Setup mock M1 dataset with gaps, duplicates, and rounding errors
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # 15 minutes of candles with:
    # - A duplicate at index 2
    # - A gap at index 8 (missing 3 minutes)
    # - An invalid price boundary at index 5 (High < Open)
    times = []
    curr = now - datetime.timedelta(minutes=20)
    for i in range(15):
        if i == 2:
            # Duplicate timestamp
            times.append(times[-1])
        elif i == 8:
            # Missing 3 minutes gap
            curr += datetime.timedelta(minutes=3)
            times.append(curr)
        else:
            curr += datetime.timedelta(minutes=1)
            times.append(curr)
            
    df_raw = pd.DataFrame({
        "timestamp": [t.timestamp() for t in times],
        "open": [2350.0] * 15,
        "high": [2355.0] * 15,
        "low": [2345.0] * 15,
        "close": [2352.0] * 15,
        "volume": [1.5] * 15
    })
    
    # Inject invalid OHLC pricing bound (High < Open)
    df_raw.loc[5, "high"] = 2340.0
    
    # 2. Run Data Cleaner
    cleaner = DataCleaner(max_spike_pct=0.02)
    df_clean, report = cleaner.clean_m1_data(df_raw)
    
    # Check cleaning assertions
    assert len(df_clean) > 0
    assert report["duplicates_removed"] == 1
    assert report["missing_bars_filled"] > 0
    
    # Validate no duplicates
    assert df_clean["timestamp"].duplicated().sum() == 0
    
    # Validate strict chronological order
    assert df_clean["timestamp"].is_monotonic_increasing
    
    # Validate no negative prices
    assert (df_clean["open"] > 0).all()
    assert (df_clean["high"] > 0).all()
    assert (df_clean["low"] > 0).all()
    assert (df_clean["close"] > 0).all()
    
    # Validate repaired price consistency bounds
    assert (df_clean["high"] >= np.maximum(df_clean["open"], df_clean["close"])).all()
    assert (df_clean["low"] <= np.minimum(df_clean["open"], df_clean["close"])).all()
    
    # 3. Run Timeframe Resampler
    generator = TimeframeGenerator(processed_dir="tests/test_processed")
    tf_summary = generator.aggregate_and_save(df_clean, symbol="TEST_GOLD")
    
    # Assert resampled files were written
    assert "M1" in tf_summary
    assert "M5" in tf_summary
    assert os.path.exists(tf_summary["M5"]["path"])
    
    df_m5 = pd.read_parquet(tf_summary["M5"]["path"])
    assert len(df_m5) == len(df_clean) // 5 or len(df_m5) == (len(df_clean) // 5) + 1
    
    # Clean up written parquet folders
    for tf_name in ["M1", "M5", "M15", "H1", "H4"]:
        path = os.path.join("tests/test_processed", tf_name, "TEST_GOLD.parquet")
        if os.path.exists(path):
            os.remove(path)
        dir_path = os.path.join("tests/test_processed", tf_name)
        if os.path.exists(dir_path):
            os.rmdir(dir_path)
    if os.path.exists("tests/test_processed"):
        os.rmdir("tests/test_processed")
