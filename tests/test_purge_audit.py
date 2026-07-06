import pytest
import pandas as pd
import numpy as np

def test_purge_audit_logic():
    # Helper function matching validation/purge_audit.py logic
    def check_overlap(train_labels_df, oos_labels_df, embargo_bars=10):
        train_exits = pd.to_datetime(train_labels_df["exit_time"])
        max_train_exit = train_exits.max()
        
        timeframe_minutes = 5
        embargo_delta = pd.Timedelta(minutes=embargo_bars * timeframe_minutes)
        embargo_limit = max_train_exit + embargo_delta
        
        oos_entries = pd.to_datetime(oos_labels_df["timestamp"])
        
        overlap_count = 0
        embargo_violations = 0
        
        for entry in oos_entries:
            if entry < max_train_exit:
                overlap_count += 1
            if entry < embargo_limit:
                embargo_violations += 1
                
        verdict = "PASS" if (overlap_count == 0 and embargo_violations == 0) else "FAIL"
        return overlap_count, embargo_violations, verdict

    # Scenario 1: Clean, non-overlapping splits
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min"),
        "exit_time": pd.date_range("2024-01-01 00:25:00", periods=5, freq="5min")
    })
    oos_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-04-01", periods=5, freq="5min"),
        "exit_time": pd.date_range("2026-04-01 00:25:00", periods=5, freq="5min")
    })
    
    overlaps, violations, verdict = check_overlap(train_df, oos_df)
    assert overlaps == 0
    assert violations == 0
    assert verdict == "PASS"

    # Scenario 2: Overlapping splits (leakage)
    train_df_overlap = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min"),
        "exit_time": [pd.Timestamp("2026-04-02 00:00:00")] * 5
    })
    
    overlaps_leak, violations_leak, verdict_leak = check_overlap(train_df_overlap, oos_df)
    assert overlaps_leak > 0
    assert violations_leak > 0
    assert verdict_leak == "FAIL"
