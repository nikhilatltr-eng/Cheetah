import pytest
import pandas as pd
from validation.significance_filter import compute_wilson_interval, filter_table_by_significance

def test_compute_wilson_interval():
    # 100 trades, 60% win rate
    lower, upper = compute_wilson_interval(100, 0.60)
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0
    assert lower < 0.60 < upper
    
    # 0 trades handling
    lower_zero, upper_zero = compute_wilson_interval(0, 0.50)
    assert lower_zero == 0.0
    assert upper_zero == 0.0

def test_filter_table_by_significance():
    df = pd.DataFrame([
        {"Exit Policy": "EMA21 exit", "Trade Count": 150, "Win Rate": 0.80, "Profit Factor": 5.0},
        {"Exit Policy": "EMA55 exit", "Trade Count": 45, "Win Rate": 0.90, "Profit Factor": 10.0}
    ])
    
    df_filtered = filter_table_by_significance(df, min_trades=100)
    
    # Check row 1 (150 trades >= 100)
    assert "EMA21 exit" in df_filtered.loc[0, "Exit Policy"]
    assert "80.00%" in df_filtered.loc[0, "Win Rate"]
    assert "[" in df_filtered.loc[0, "Win Rate"]  # Wilson interval check
    assert df_filtered.loc[0, "Profit Factor"] == 5.0
    
    # Check row 2 (45 trades < 100)
    assert df_filtered.loc[1, "Win Rate"] == "INSUFFICIENT DATA — NOT STATISTICALLY MEANINGFUL"
    assert df_filtered.loc[1, "Profit Factor"] == "INSUFFICIENT DATA — NOT STATISTICALLY MEANINGFUL"
