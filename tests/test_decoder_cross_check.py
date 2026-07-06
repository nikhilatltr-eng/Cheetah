import pytest
import pandas as pd
import numpy as np

def run_comparison_logic(df_ref, df_pipeline, tolerance=1e-5):
    """
    Merges reference and pipeline dataframes on timestamp and compares columns.
    Returns audit statistics and detailed mismatches.
    """
    # Standardize columns and index
    df_ref = df_ref.copy()
    df_pipeline = df_pipeline.copy()
    
    df_ref["timestamp"] = pd.to_datetime(df_ref["timestamp"])
    df_pipeline["timestamp"] = pd.to_datetime(df_pipeline["timestamp"])
    
    # Merge on timestamp
    merged = pd.merge(df_ref, df_pipeline, on="timestamp", suffixes=("_ref", "_pipe"))
    
    mismatches = []
    matches_exact = 0
    matches_tolerance = 0
    mismatched_beyond = 0
    
    cols = ["open", "high", "low", "close", "volume"]
    
    for _, row in merged.iterrows():
        ts = row["timestamp"]
        mismatch_row = {}
        has_mismatch = False
        
        for col in cols:
            val_ref = row[f"{col}_ref"]
            val_pipe = row[f"{col}_pipe"]
            
            diff = abs(val_ref - val_pipe)
            pct_diff = (diff / val_ref) * 100.0 if val_ref != 0 else 0.0
            
            if diff > tolerance:
                has_mismatch = True
                mismatch_row[col] = {
                    "ref": val_ref,
                    "pipe": val_pipe,
                    "diff": diff,
                    "pct_diff": pct_diff
                }
        
        if has_mismatch:
            mismatched_beyond += 1
            mismatches.append({
                "timestamp": ts,
                "details": mismatch_row
            })
        else:
            # Check exact vs tolerance
            is_exact = True
            for col in cols:
                if row[f"{col}_ref"] != row[f"{col}_pipe"]:
                    is_exact = False
                    break
            if is_exact:
                matches_exact += 1
            else:
                matches_tolerance += 1
                
    total_compared = len(merged)
    
    # Check missing timestamps
    ref_ts = set(df_ref["timestamp"])
    pipe_ts = set(df_pipeline["timestamp"])
    
    missing_in_pipe = sorted(list(ref_ts - pipe_ts))
    missing_in_ref = sorted(list(pipe_ts - ref_ts))
    
    return {
        "total_compared": total_compared,
        "matches_exact": matches_exact,
        "matches_tolerance": matches_tolerance,
        "mismatches_beyond_tolerance": mismatched_beyond,
        "missing_in_pipeline": missing_in_pipe,
        "missing_in_reference": missing_in_ref,
        "mismatches": mismatches
    }

def test_decoder_comparison_exact_match():
    # Setup identical datasets
    timestamps = pd.date_range("2026-04-22 00:00:00", periods=5, freq="1min")
    df_ref = pd.DataFrame({
        "timestamp": timestamps,
        "open": [10.0, 11.0, 12.0, 13.0, 14.0],
        "high": [10.5, 11.5, 12.5, 13.5, 14.5],
        "low": [9.5, 10.5, 11.5, 12.5, 13.5],
        "close": [10.2, 11.2, 12.2, 13.2, 14.2],
        "volume": [1.0, 2.0, 1.5, 2.5, 3.0]
    })
    df_pipe = df_ref.copy()
    
    res = run_comparison_logic(df_ref, df_pipe)
    assert res["total_compared"] == 5
    assert res["matches_exact"] == 5
    assert res["mismatches_beyond_tolerance"] == 0
    assert len(res["missing_in_pipeline"]) == 0

def test_decoder_comparison_mismatch():
    timestamps = pd.date_range("2026-04-22 00:00:00", periods=5, freq="1min")
    df_ref = pd.DataFrame({
        "timestamp": timestamps,
        "open": [10.0, 11.0, 12.0, 13.0, 14.0],
        "high": [10.5, 11.5, 12.5, 13.5, 14.5],
        "low": [9.5, 10.5, 11.5, 12.5, 13.5],
        "close": [10.2, 11.2, 12.2, 13.2, 14.2],
        "volume": [1.0, 2.0, 1.5, 2.5, 3.0]
    })
    
    # Deliberate corruption: swap high and low in row index 2
    df_pipe = df_ref.copy()
    df_pipe.loc[2, "high"] = 11.5
    df_pipe.loc[2, "low"] = 12.5
    
    res = run_comparison_logic(df_ref, df_pipe)
    assert res["total_compared"] == 5
    assert res["matches_exact"] == 4
    assert res["mismatches_beyond_tolerance"] == 1
    assert len(res["mismatches"]) == 1
    assert res["mismatches"][0]["timestamp"] == pd.Timestamp("2026-04-22 00:02:00")
