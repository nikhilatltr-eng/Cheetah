import math
import pandas as pd
import numpy as np

def compute_wilson_interval(n_trades: int, win_rate: float, conf_level: float = 0.95) -> tuple:
    """
    Computes the 95% Wilson score interval for a binomial proportion.
    Returns:
        (lower_bound, upper_bound): tuple of floats bounded between 0.0 and 1.0.
    """
    if n_trades <= 0:
        return 0.0, 0.0
    
    # 95% confidence level corresponds to z = 1.96
    z = 1.96
    p = win_rate
    
    denominator = 1.0 + (z**2 / n_trades)
    center = (p + (z**2 / (2.0 * n_trades))) / denominator
    spread = z * math.sqrt((p * (1.0 - p) / n_trades) + (z**2 / (4.0 * n_trades**2))) / denominator
    
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return lower, upper

def filter_table_by_significance(df: pd.DataFrame, min_trades: int = 100, trade_count_col: str = "Trade Count") -> pd.DataFrame:
    """
    Applies the significance gate to a DataFrame table.
    Any row where trade count < min_trades gets its metric columns overwritten
    with an 'INSUFFICIENT DATA' message.
    """
    df_filtered = df.copy()
    
    # Identify metric columns to blank out
    metric_cols = [c for c in df.columns if c != trade_count_col and c != "Exit Policy" and c != "Session" and c != "Cost Level" and c != "Month" and c != "Entry Hours (UTC)"]
    
    # Ensure columns exist in string representation to accommodate messages
    for col in metric_cols:
        df_filtered[col] = df_filtered[col].astype(object)
        
    for idx, row in df.iterrows():
        n = int(row[trade_count_col])
        if n < min_trades:
            for col in metric_cols:
                df_filtered.at[idx, col] = "INSUFFICIENT DATA — NOT STATISTICALLY MEANINGFUL"
        else:
            # If Win Rate column is present, append the Wilson interval to it
            if "Win Rate" in row:
                wr = float(row["Win Rate"])
                # Handle potential string percent format
                if isinstance(row["Win Rate"], str) and row["Win Rate"].endswith("%"):
                    wr = float(row["Win Rate"].replace("%", "")) / 100.0
                
                lower, upper = compute_wilson_interval(n, wr)
                wr_formatted = f"{wr:.2%}" if not isinstance(row["Win Rate"], str) else row["Win Rate"]
                df_filtered.at[idx, "Win Rate"] = f"{wr_formatted} [{lower:.1%} - {upper:.1%}]"
                
    return df_filtered
