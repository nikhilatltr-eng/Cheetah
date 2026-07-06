import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features import (
    compute_ema_stack,
    compute_adx,
    compute_macd,
    compute_atr,
    compute_rsi,
    compute_swing_points,
    compute_liquidity_sweeps,
    compute_all_features
)

def test_flat_price_features():
    """Tests feature engineering behavior when price is completely flat."""
    n_bars = 250
    timestamps = pd.date_range(start="2026-07-05", periods=n_bars, freq="min", tz="UTC")
    
    # Flat price at 2000.0
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [2000.0] * n_bars,
        "high": [2000.0] * n_bars,
        "low": [2000.0] * n_bars,
        "close": [2000.0] * n_bars,
        "tick_volume": [10.0] * n_bars,
        "spread": [2.0] * n_bars,
        "real_volume": [100.0] * n_bars
    })
    
    df_feat = compute_all_features(df)
    
    # Assert EMA stack is warmed up and equals the close price
    assert df_feat["ema_8"].iloc[-1] == 2000.0
    assert df_feat["ema_21"].iloc[-1] == 2000.0
    assert df_feat["ema_55"].iloc[-1] == 2000.0
    assert df_feat["ema_200"].iloc[-1] == 2000.0
    
    # ATR should be 0 since high == low == close
    assert df_feat["atr_14"].iloc[-1] == 0.0
    assert df_feat["atr_21"].iloc[-1] == 0.0
    assert df_feat["atr_55"].iloc[-1] == 0.0
    
    # RSI should be 50 when flat (no change)
    assert df_feat["rsi"].iloc[-1] == 50.0
    
    # Realized volatility should be 0 (no returns)
    assert df_feat["realized_vol"].iloc[-1] == 0.0
    
    # No swings should be found on a flat line
    assert not df_feat["swing_high"].any()
    assert not df_feat["swing_low"].any()

def test_hand_computed_ema():
    """Verify EMA computation on a simple moving sequence."""
    # EMA formula: EMA_t = Close_t * alpha + EMA_{t-1} * (1 - alpha)
    # where alpha = 2 / (N + 1). For N=8, alpha = 2 / 9 = 0.222222...
    closes = [10.0, 12.0, 11.0, 13.0, 12.0]
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-05", periods=5, freq="min"),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "tick_volume": [1.0]*5, "spread": [1.0]*5, "real_volume": [1.0]*5
    })
    
    df = compute_ema_stack(df)
    
    # Hand calculations:
    # ema_8[0] = 10.0
    # ema_8[1] = 12.0 * (2/9) + 10.0 * (7/9) = 2.66667 + 7.77778 = 10.44444
    # ema_8[2] = 11.0 * (2/9) + 10.44444 * (7/9) = 2.44444 + 8.12345 = 10.5679
    
    assert pytest.approx(df["ema_8"].iloc[0], 0.0001) == 10.0
    assert pytest.approx(df["ema_8"].iloc[1], 0.0001) == 10.4444
    assert pytest.approx(df["ema_8"].iloc[2], 0.0001) == 10.5679

def test_hand_computed_atr():
    """Verify ATR computation on a moving sequence."""
    # Series of 3 bars
    # Bar 0: O:10, H:12, L:9, C:11  -> TR = H-L = 3
    # Bar 1: O:11, H:13, L:10, C:12 -> TR = max(13-10, 13-11, 11-10) = 3
    # Bar 2: O:12, H:15, L:11, C:13 -> TR = max(15-11, 15-12, 12-11) = 4
    # For ATR 14: alpha = 1 / 14
    # atr[0] = TR[0] = 3.0
    # atr[1] = 3.0 * (1/14) + 3.0 * (13/14) = 3.0
    # atr[2] = 4.0 * (1/14) + 3.0 * (13/14) = 0.2857 + 2.7857 = 3.0714
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-05", periods=3, freq="min"),
        "open": [10.0, 11.0, 12.0],
        "high": [12.0, 13.0, 15.0],
        "low": [9.0, 10.0, 11.0],
        "close": [11.0, 12.0, 13.0],
        "tick_volume": [1.0]*3, "spread": [1.0]*3, "real_volume": [1.0]*3
    })
    
    df = compute_atr(df)
    assert pytest.approx(df["atr_14"].iloc[0], 0.0001) == 3.0
    assert pytest.approx(df["atr_14"].iloc[1], 0.0001) == 3.0
    assert pytest.approx(df["atr_14"].iloc[2], 0.0001) == 3.0714

def test_rsi():
    """Verify RSI calculation on standard values."""
    # Prices: 10, 12, 11, 13
    # Changes: +2, -1, +2
    # alpha = 1/14
    # Gains: 0, 2, 0, 2
    # Losses: 0, 0, 1, 0
    # avg_gain[0] = 0; avg_loss[0] = 0
    # avg_gain[1] = 2 * (1/14) = 0.142857; avg_loss[1] = 0
    # avg_gain[2] = 0.142857 * (13/14) = 0.132653; avg_loss[2] = 1 * (1/14) = 0.071428
    # avg_gain[3] = 0.132653 * (13/14) + 2 * (1/14) = 0.123178 + 0.142857 = 0.266035
    # avg_loss[3] = 0.071428 * (13/14) = 0.066326
    # rs[3] = 0.266035 / 0.066326 = 4.011
    # rsi[3] = 100 - 100/(1 + 4.011) = 100 - 19.95 = 80.05
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-05", periods=4, freq="min"),
        "open": [10.0, 12.0, 11.0, 13.0],
        "high": [10.0, 12.0, 11.0, 13.0],
        "low": [10.0, 12.0, 11.0, 13.0],
        "close": [10.0, 12.0, 11.0, 13.0],
        "tick_volume": [1.0]*4, "spread": [1.0]*4, "real_volume": [1.0]*4
    })
    
    df = compute_rsi(df)
    assert pytest.approx(df["rsi"].iloc[3], 0.05) == 80.05

def test_swing_points_and_sweeps():
    """Verify swing highs and lows and liquidity sweeps on synthetic fractal patterns."""
    # Index:  0   1   2   3   4   5   6   7   8   9
    highs =  [10, 11, 13, 11, 10, 10, 10, 10, 15, 12]
    lows =   [9,  8,  6,  8,  9,  9,  9,  9,  7,  8]
    closes = [9.5, 9, 10, 9.5, 9.5, 9.5, 9.5, 9.5, 11, 10]
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-05", periods=10, freq="min"),
        "open": closes, "high": highs, "low": lows, "close": closes,
        "tick_volume": [1.0]*10, "spread": [1.0]*10, "real_volume": [1.0]*10
    })
    
    df = compute_swing_points(df)
    
    # Index 2 high=13: higher than 10,11 (prev 2) and 11,10 (next 2). Should be swing high.
    # Index 2 low=6: lower than 9,8 (prev 2) and 8,9 (next 2). Should be swing low.
    assert bool(df["swing_high"].iloc[2]) is True
    assert bool(df["swing_low"].iloc[2]) is True
    
    # Assert other indices are False
    assert bool(df["swing_high"].iloc[3]) is False
    assert bool(df["swing_low"].iloc[3]) is False

    # Test Liquidity Sweep
    # Let's create a scenario where we have a swing low at index 2 (low = 6)
    # and at index 8 we spike down to low = 5.5 but close = 8 (which is > swing low of 6)
    # Index:  0   1   2   3   4   5   6   7   8   9
    highs =  [10, 11, 13, 11, 10, 10, 10, 10, 12, 10]
    lows =   [9,  8,  6,  8,  9,  9,  9,  9,  5.5, 8]
    closes = [9.5, 9, 10, 9.5, 9.5, 9.5, 9.5, 9.5, 8.0, 9]
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-05", periods=10, freq="min"),
        "open": closes, "high": highs, "low": lows, "close": closes,
        "tick_volume": [1.0]*10, "spread": [1.0]*10, "real_volume": [1.0]*10
    })
    
    df = compute_liquidity_sweeps(df)
    # At index 8:
    # recent_swing_low from index 2 was 6.0.
    # Current low (5.5) < 6.0
    # Current close (8.0) > 6.0
    # This represents a bullish sweep!
    assert bool(df["liquidity_sweep_bull"].iloc[8]) is True
    assert bool(df["liquidity_sweep"].iloc[8]) is True
