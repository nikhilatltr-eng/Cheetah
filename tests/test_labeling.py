import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from labeling import get_triple_barrier_labels, get_sample_weights

def test_triple_barrier_stop_loss():
    """Verify that stop loss is hit correctly on a downward price spike."""
    dates = pd.date_range("2026-07-05", periods=5, freq="min", tz="UTC")
    # pt_mult=2.0, sl_mult=1.0. Target: +4.0, Stop: -2.0 (ATR=2.0)
    df = pd.DataFrame({
        "timestamp": dates,
        "open":  [100.0, 101.0, 97.5, 96.0, 95.0],
        "high":  [100.0, 101.0, 97.5, 96.0, 95.0],
        "low":   [100.0, 101.0, 97.5, 96.0, 95.0],
        "close": [100.0, 101.0, 97.5, 96.0, 95.0],
        "atr_14": [2.0] * 5
    })
    
    res = get_triple_barrier_labels(df, atr_col="atr_14", pt_mult=2.0, sl_mult=1.0, max_holding_bars=4)
    
    # Row 0: price = 100.0. Stop is at 98.0.
    # At index 2: price/low hits 97.5 which is <= 98.0.
    assert res.iloc[0]["label"] == -1
    assert res.iloc[0]["exit_time"] == dates[2]
    assert res.iloc[0]["holding_bars"] == 2

def test_triple_barrier_profit_target():
    """Verify that profit target is hit correctly on an upward price spike."""
    dates = pd.date_range("2026-07-05", periods=5, freq="min", tz="UTC")
    # pt_mult=2.0, sl_mult=1.0. Target: +4.0, Stop: -2.0 (ATR=2.0)
    df = pd.DataFrame({
        "timestamp": dates,
        "open":  [100.0, 102.0, 104.5, 105.0, 106.0],
        "high":  [100.0, 102.0, 104.5, 105.0, 106.0],
        "low":   [100.0, 102.0, 104.5, 105.0, 106.0],
        "close": [100.0, 102.0, 104.5, 105.0, 106.0],
        "atr_14": [2.0] * 5
    })
    
    res = get_triple_barrier_labels(df, atr_col="atr_14", pt_mult=2.0, sl_mult=1.0, max_holding_bars=4)
    
    # Row 0: price = 100.0. Profit target is at 104.0.
    # At index 2: price/high hits 104.5 which is >= 104.0.
    assert res.iloc[0]["label"] == 1
    assert res.iloc[0]["exit_time"] == dates[2]
    assert res.iloc[0]["holding_bars"] == 2

def test_triple_barrier_timeout():
    """Verify that vertical barrier timeout acts correctly when price stays flat."""
    dates = pd.date_range("2026-07-05", periods=10, freq="min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "open":  [100.0] * 10,
        "high":  [100.0] * 10,
        "low":   [100.0] * 10,
        "close": [100.0] * 10,
        "atr_14": [2.0] * 10
    })
    
    res = get_triple_barrier_labels(df, atr_col="atr_14", pt_mult=2.0, sl_mult=1.0, max_holding_bars=3)
    
    # Max holding bars is 3. Row 0 exit must be at index 3.
    assert res.iloc[0]["label"] == 0
    assert res.iloc[0]["exit_time"] == dates[3]
    assert res.iloc[0]["holding_bars"] == 3

def test_uniqueness_sample_weights():
    """Verify that overlapping spans have lower sample weights than non-overlapping spans."""
    dates = pd.date_range("2026-07-05", periods=6, freq="min", tz="UTC")
    # Let's mock a labels dataframe where:
    # Row 0: active from index 0 to 4 (overlap with row 1)
    # Row 1: active from index 1 to 2 (overlap with row 0)
    # Row 5: active from index 5 to 5 (no overlap)
    labels_df = pd.DataFrame({
        "label": [1.0, 1.0, np.nan, np.nan, np.nan, 1.0],
        "holding_bars": [4.0, 1.0, np.nan, np.nan, np.nan, 0.0]
    }, index=dates)
    
    weights = get_sample_weights(labels_df)
    
    # Weights should be normalized
    assert pytest.approx(weights.mean(), 0.0001) == 1.0
    
    # Row 5 has no overlaps, should have the highest uniqueness (hence highest weight)
    # compared to Row 0 and Row 1 which overlap on index 1 and 2
    assert weights.iloc[5] > weights.iloc[0]
    assert weights.iloc[5] > weights.iloc[1]
