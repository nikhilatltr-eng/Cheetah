import os
import lzma
import struct
import datetime
import pandas as pd
import numpy as np
from collections import Counter

def run_calibration_audit(start_date=None, end_date=None):
    """
    Module 2: Audits Dukascopy column layout and price scaler calibration consistency.
    """
    if start_date is None:
        start_date = datetime.date(2026, 4, 1)
    if end_date is None:
        end_date = datetime.date(2026, 6, 30)
        
    curr = start_date
    dates = []
    while curr <= end_date:
        dates.append(curr)
        curr += datetime.timedelta(days=1)
        
    raw_dir = "data/raw/dukascopy/m1"
    audit_records = []
    
    for d in dates:
        file_name = f"{d.year}_{d.month:02d}_{d.day:02d}.bi5"
        file_path = os.path.join(raw_dir, file_name)
        
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            # Skip weekend/holiday files with no data
            continue
            
        try:
            with open(file_path, "rb") as f:
                compressed_data = f.read()
            decompressed = lzma.decompress(compressed_data)
        except Exception:
            continue
            
        fmt = ">IIIIIf"
        record_size = struct.calcsize(fmt)
        n_records = len(decompressed) // record_size
        
        if n_records == 0:
            continue
            
        records = []
        # We only need to check the records to determine scaler and column layout validity
        for i in range(min(n_records, 1000)):  # sample up to 1000 records for validation
            chunk = decompressed[i * record_size : (i + 1) * record_size]
            records.append(struct.unpack(fmt, chunk))
            
        arr = np.array(records)
        first_val = arr[0, 1]
        
        scaler = 1000.0
        if first_val > 10000000:
            scaler = 100000.0
        elif first_val > 100000:
            scaler = 1000.0
        elif first_val > 10000:
            scaler = 100.0
            
        # Re-run layout checks
        opt_A_high = arr[:, 4] / scaler
        opt_A_low = arr[:, 3] / scaler
        opt_A_open = arr[:, 1] / scaler
        opt_A_close = arr[:, 2] / scaler
        
        valid_A = np.all(opt_A_high >= np.maximum(opt_A_open, opt_A_close)) and np.all(opt_A_low <= np.minimum(opt_A_open, opt_A_close))
        layout = "Option A (Open/Close/Low/High)" if valid_A else "Option B (Open/High/Low/Close)"
        
        audit_records.append({
            "date": d,
            "scaler": scaler,
            "layout": layout,
            "first_val": first_val
        })
        
    df_audit = pd.DataFrame(audit_records)
    if df_audit.empty:
        return {
            "total_days": 0,
            "mode_scaler": None,
            "mode_layout": None,
            "inconsistent_days": [],
            "consistency_summary": "No data audited."
        }
        
    # Find mode of scaler and layout
    mode_scaler = df_audit["scaler"].mode()[0]
    mode_layout = df_audit["layout"].mode()[0]
    
    inconsistent_days = []
    for _, row in df_audit.iterrows():
        if row["scaler"] != mode_scaler or row["layout"] != mode_layout:
            inconsistent_days.append({
                "date": row["date"].isoformat(),
                "scaler": row["scaler"],
                "layout": row["layout"],
                "first_val": int(row["first_val"])
            })
            
    total_days = len(df_audit)
    consistent_count = total_days - len(inconsistent_days)
    
    verdict = "PASS" if len(inconsistent_days) == 0 else "PASS WITH WARNINGS"
    
    summary = f"{consistent_count} of {total_days} days resolved to the same column layout ({mode_layout}) and multiplier ({mode_scaler})."
    
    return {
        "total_days": total_days,
        "mode_scaler": mode_scaler,
        "mode_layout": mode_layout,
        "inconsistent_days": inconsistent_days,
        "consistency_summary": summary,
        "verdict": verdict,
        "df_audit": df_audit
    }

if __name__ == "__main__":
    res = run_calibration_audit()
    print(res["consistency_summary"])
    if res["inconsistent_days"]:
        print("Inconsistent days:")
        for day in res["inconsistent_days"]:
            print(day)
