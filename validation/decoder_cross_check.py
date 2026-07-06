import os
import datetime
import random
import pandas as pd
import numpy as np
from data.dukascopy_downloader import DukascopyDownloader

try:
    import dukascopy_python
except ImportError:
    dukascopy_python = None

def run_cross_check(dates=None, tolerance=1e-5):
    """
    Module 1: Cross-checks our pipeline's decoder output against dukascopy-python reference.
    """
    if dukascopy_python is None:
        raise ImportError("dukascopy-python package is not installed.")
        
    if dates is None:
        # Default target dates + 1 random day from April, May, June 2026
        random.seed(42)
        # Target dates
        target_dates = [
            datetime.date(2026, 4, 22),
            datetime.date(2026, 4, 23)
        ]
        # Random dates (avoiding weekends where there might be 404s/no data)
        # Weekdays in April: 15 (Wed)
        # Weekdays in May: 12 (Tue)
        # Weekdays in June: 18 (Thu)
        target_dates.append(datetime.date(2026, 4, 15))
        target_dates.append(datetime.date(2026, 5, 12))
        target_dates.append(datetime.date(2026, 6, 18))
        dates = sorted(list(set(target_dates)))

    downloader = DukascopyDownloader(symbol="XAUUSD", raw_dir="data/raw/dukascopy")
    os.makedirs("data/processed/M1", exist_ok=True)
    
    results = {}
    
    for d in dates:
        # 1. Prepare pipeline decoded file in data/processed/M1/
        filename = f"{d.year}_{d.month:02d}_{d.day:02d}.parquet"
        processed_path = os.path.join("data/processed/M1", filename)
        
        if not os.path.exists(processed_path):
            # Download raw .bi5 file
            raw_path = downloader.download_day(d)
            if raw_path:
                df_day = downloader.parse_bi5(raw_path, d)
                if not df_day.empty:
                    df_day.to_parquet(processed_path)
                    
        # 2. Load pipeline decoded data
        if os.path.exists(processed_path):
            df_pipe = pd.read_parquet(processed_path)
        else:
            df_pipe = pd.DataFrame()
            
        # 3. Fetch reference data using dukascopy_python
        start_dt = datetime.datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end_dt = datetime.datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=datetime.timezone.utc)
        
        try:
            df_ref_raw = dukascopy_python.fetch(
                instrument="XAU/USD",
                interval=dukascopy_python.INTERVAL_MIN_1,
                offer_side="BID",
                start=start_dt,
                end=end_dt
            )
            if df_ref_raw is not None and not df_ref_raw.empty:
                df_ref = df_ref_raw.reset_index()
                # Ensure timestamp is timezone-aware UTC
                df_ref["timestamp"] = pd.to_datetime(df_ref["timestamp"])
            else:
                df_ref = pd.DataFrame()
        except Exception as e:
            print(f"Error fetching from reference dukascopy-python for {d}: {e}")
            df_ref = pd.DataFrame()
            
        # 4. Perform cross comparison
        if df_pipe.empty or df_ref.empty:
            results[d] = {
                "total_compared": 0,
                "matches_exact": 0,
                "matches_tolerance": 0,
                "mismatches_beyond_tolerance": 0,
                "missing_in_pipeline": [],
                "missing_in_reference": [],
                "mismatches": [],
                "error": "Missing pipeline or reference data for this date."
            }
            continue
            
        df_pipe["timestamp"] = pd.to_datetime(df_pipe["timestamp"])
        
        # Merge on timestamp
        merged = pd.merge(df_ref, df_pipe, on="timestamp", suffixes=("_ref", "_pipe"))
        
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
                    "timestamp": str(ts),
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
                    
        # Check missing timestamps
        ref_ts = set(df_ref["timestamp"])
        pipe_ts = set(df_pipe["timestamp"])
        
        missing_in_pipe = [str(t) for t in sorted(list(ref_ts - pipe_ts))]
        missing_in_ref = [str(t) for t in sorted(list(pipe_ts - ref_ts))]
        
        results[d] = {
            "total_compared": len(merged),
            "matches_exact": matches_exact,
            "matches_tolerance": matches_tolerance,
            "mismatches_beyond_tolerance": mismatched_beyond,
            "missing_in_pipeline": missing_in_pipe,
            "missing_in_reference": missing_in_ref,
            "mismatches": mismatches
        }
        
    return results

if __name__ == "__main__":
    res = run_cross_check()
    for d, info in res.items():
        print(f"Date: {d} | Compared: {info['total_compared']} | Exact: {info['matches_exact']} | Beyond: {info['mismatches_beyond_tolerance']}")
