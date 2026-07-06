import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def get_triple_barrier_labels(df: pd.DataFrame, atr_col: str = "atr_14", pt_mult: float = 2.0, sl_mult: float = 2.0, max_holding_bars: int = 20) -> pd.DataFrame:
    """
    Computes triple-barrier labels (profit target, stop loss, and vertical timeout barriers)
    inspired by Lopez de Prado.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing columns ['timestamp', 'open', 'high', 'low', 'close'] and the specified atr_col.
        atr_col (str): Column name for the ATR indicator to use.
        pt_mult (float): Multiplier for the profit target upper barrier.
        sl_mult (float): Multiplier for the stop loss lower barrier.
        max_holding_bars (int): Number of bars representing the vertical barrier (timeout).
        
    Returns:
        pd.DataFrame: Contains columns:
            - 'entry_time': timestamp of entry
            - 'exit_time': timestamp when any barrier is first breached
            - 'exit_price': close price at barrier breach
            - 'label': 1 (profit target hit), -1 (stop loss hit), 0 (timed out)
            - 'holding_bars': number of bars held
    """
    n = len(df)
    entry_times = []
    exit_times = []
    exit_prices = []
    labels = []
    holding_bars_list = []
    
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    timestamp = df["timestamp"].values
    
    # Handle missing ATR column fallback
    if atr_col not in df.columns:
        logger.warning(f"ATR column '{atr_col}' not found. Using rolling standard deviation as proxy.")
        # Calculate rolling std of close as a quick proxy
        atr = df["close"].diff().rolling(14).std().fillna(0.0).values
    else:
        atr = df[atr_col].values
        
    for i in range(n):
        entry_t = timestamp[i]
        entry_p = close[i]
        atr_val = atr[i]
        
        if pd.isna(atr_val) or atr_val <= 0:
            # Insufficient warmup data, label as NaN
            entry_times.append(entry_t)
            exit_times.append(pd.NaT)
            exit_prices.append(np.nan)
            labels.append(np.nan)
            holding_bars_list.append(np.nan)
            continue
            
        upper_barrier = entry_p + pt_mult * atr_val
        lower_barrier = entry_p - sl_mult * atr_val
        
        exit_idx = None
        label = 0
        
        # Look forward up to max_holding_bars
        limit = min(i + max_holding_bars, n - 1)
        for j in range(i + 1, limit + 1):
            h_val = high[j]
            l_val = low[j]
            
            hit_upper = h_val >= upper_barrier
            hit_lower = l_val <= lower_barrier
            
            if hit_upper and hit_lower:
                # Double barrier breach on same bar: classify based on close direction
                if close[j] >= entry_p:
                    label = 1
                else:
                    label = -1
                exit_idx = j
                break
            elif hit_upper:
                label = 1
                exit_idx = j
                break
            elif hit_lower:
                label = -1
                exit_idx = j
                break
                
        if exit_idx is None:
            # Vertical barrier (timeout) hit
            exit_idx = limit
            label = 0
            
        entry_times.append(entry_t)
        exit_times.append(timestamp[exit_idx])
        exit_prices.append(close[exit_idx])
        labels.append(label)
        holding_bars_list.append(exit_idx - i)
        
    res = pd.DataFrame({
        "entry_time": entry_times,
        "exit_time": exit_times,
        "exit_price": exit_prices,
        "label": labels,
        "holding_bars": holding_bars_list
    }, index=df.index)
    
    # Ensure exit_time has tz-aware datetime formats matching input
    res["entry_time"] = pd.to_datetime(res["entry_time"], utc=True)
    res["exit_time"] = pd.to_datetime(res["exit_time"], utc=True)
    
    return res

def get_sample_weights(labels_df: pd.DataFrame) -> pd.Series:
    """
    Computes sample weights based on average uniqueness of label spans
    to penalize overlapping labels (concurrency downweighting).
    """
    n = len(labels_df)
    if n == 0:
        return pd.Series(dtype=float)
        
    # Vectorized array to keep count of active labels at each timestamp index
    concurrency = np.zeros(n)
    
    for i in range(n):
        row = labels_df.iloc[i]
        if pd.isna(row["label"]) or pd.isna(row["holding_bars"]):
            continue
            
        holding = int(row["holding_bars"])
        j = min(i + holding, n - 1)
        concurrency[i:j+1] += 1
        
    # Clip to avoid division by zero
    concurrency = np.where(concurrency == 0, 1.0, concurrency)
    
    # Uniqueness at each point in time
    uniqueness = 1.0 / concurrency
    
    # Average uniqueness per sample span
    weights = []
    for i in range(n):
        row = labels_df.iloc[i]
        if pd.isna(row["label"]) or pd.isna(row["holding_bars"]):
            weights.append(1.0)
            continue
            
        holding = int(row["holding_bars"])
        j = min(i + holding, n - 1)
        avg_uniq = np.mean(uniqueness[i:j+1])
        weights.append(avg_uniq)
        
    weights = np.array(weights)
    weights = np.nan_to_num(weights, nan=1.0)
    
    # Normalize weights so that their mean equals 1.0
    mean_w = np.mean(weights)
    if mean_w > 0:
        weights = weights / mean_w
        
    return pd.Series(weights, index=labels_df.index)
