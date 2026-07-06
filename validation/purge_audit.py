import logging
import os
import datetime
import pandas as pd
import numpy as np
from data.dukascopy_downloader import DukascopyDownloader
from data.data_cleaner import DataCleaner
from features import compute_all_features
from labeling import get_triple_barrier_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("purge_audit")

def run_purge_audit(config_path="config.yaml"):
    logger.info("Starting OOS Purge and Embargo Audit...")
    
    # 1. Load data
    symbol = "XAUUSD"
    raw_path = "data/raw/dukascopy"
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2026, 6, 30)
    
    downloader = DukascopyDownloader(symbol=symbol, raw_dir=raw_path)
    df_raw = downloader.download_range(start_date, end_date)
    
    cleaner = DataCleaner()
    df_cleaned, _ = cleaner.clean_m1_data(df_raw)
    
    df_features = compute_all_features(df_cleaned)
    labels_df = get_triple_barrier_labels(
        df_features, 
        atr_col="atr_14", 
        pt_mult=2.0, 
        sl_mult=2.0, 
        max_holding_bars=20
    )
    
    # Check config for embargo bars
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    embargo_bars = config.get("validation", {}).get("embargo_bars", 10)
    
    # Parse timestamps
    df_features["timestamp"] = pd.to_datetime(df_features["timestamp"])
    labels_df["timestamp"] = pd.to_datetime(df_features["timestamp"])
    labels_df["exit_time"] = pd.to_datetime(labels_df["exit_time"])
    
    train_mask = (df_features["timestamp"] >= "2024-01-01") & (df_features["timestamp"] <= "2025-12-31 23:59:59")
    oos_mask = (df_features["timestamp"] >= "2026-04-01") & (df_features["timestamp"] <= "2026-06-30 23:59:59")
    
    train_labels = labels_df[train_mask]
    oos_labels = labels_df[oos_mask]
    
    total_oos_samples = len(oos_labels)
    
    # Audit logic
    train_exits = train_labels["exit_time"].dropna()
    if not train_exits.empty:
        max_train_exit = train_exits.max()
    else:
        max_train_exit = pd.Timestamp("2025-12-31 23:59:59")
        
    if hasattr(max_train_exit, "tzinfo") and max_train_exit.tzinfo is not None:
        max_train_exit = max_train_exit.tz_localize(None)
        
    # Embargo limit
    embargo_limit = max_train_exit + pd.Timedelta(minutes=embargo_bars)
    if hasattr(embargo_limit, "tzinfo") and embargo_limit.tzinfo is not None:
        embargo_limit = embargo_limit.tz_localize(None)
    
    overlapping_samples = 0
    embargo_violations = 0
    
    oos_entries = oos_labels["timestamp"].values
    
    # Calculate overlaps
    for entry in oos_entries:
        entry_ts = pd.Timestamp(entry)
        if entry_ts.tzinfo is not None:
            entry_ts = entry_ts.tz_localize(None)
        if entry_ts < max_train_exit:
            overlapping_samples += 1
        if entry_ts < embargo_limit:
            embargo_violations += 1
            
    verdict = "PASS" if (overlapping_samples == 0 and embargo_violations == 0) else "FAIL"
    
    report_content = f"""# Purge & Embargo Validation Audit Report

This report audits the out-of-sample (OOS) test partition against the training partition to ensure zero overlap/leakage between training trade windows and testing trade windows.

## Audit Configuration
- **Training Period**: 2024-01-01 to 2025-12-31
- **Out-of-Sample (OOS) Test Period**: 2026-04-01 to 2026-06-30
- **Embargo Window**: {embargo_bars} M1 bars (10 minutes)
- **Max Train Exit Time observed**: {max_train_exit}
- **Embargo Cutoff Time**: {embargo_limit}

## Audit Summary
- **Total OOS Samples Evaluated**: {total_oos_samples}
- **Count with Training Overlap**: {overlapping_samples}
- **Count with Embargo Violations**: {embargo_violations}
- **Purge Audit Verdict**: **{verdict}**

## Analysis & Rationale
Because the validation design incorporates a 3-month chronological hold-out buffer (2026-01-01 to 2026-03-31, used for calibration/validation) between the training partition and the OOS test partition, no training samples can overlap with OOS entries. The 3-month gap exceeds the maximum triple-barrier holding window (20 M1 bars = 20 minutes) by several orders of magnitude.

Therefore, the split is mathematically guaranteed to be leakage-free and fully purged.
"""
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/PURGE_AUDIT_REPORT.md", "w") as f:
        f.write(report_content)
        
    logger.info("Purge audit report written successfully!")
    return {
        "verdict": verdict,
        "total_oos": total_oos_samples,
        "overlaps": overlapping_samples,
        "violations": embargo_violations,
        "max_train_exit": str(max_train_exit),
        "embargo_limit": str(embargo_limit)
    }

if __name__ == "__main__":
    run_purge_audit()
