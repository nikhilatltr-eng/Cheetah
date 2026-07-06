import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def compute_ema_stack(df: pd.DataFrame) -> pd.DataFrame:
    """Computes EMA stack (8, 21, 55, 200)."""
    df["ema_8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_55"] = df["close"].ewm(span=55, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    return df

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Computes ADX and DI+ / DI- using Wilder's smoothing."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # True Range (TR)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement (DM)
    plus_dm = high - prev_high
    minus_dm = prev_low - low
    
    # +DM / -DM logic
    plus_dm_val = np.where((plus_dm > 0) & (plus_dm > minus_dm), plus_dm, 0.0)
    minus_dm_val = np.where((minus_dm > 0) & (minus_dm > plus_dm), minus_dm, 0.0)
    
    # Wilder's Smoothing (EMA with alpha = 1 / period)
    tr_smoothed = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_di_smoothed = pd.Series(plus_dm_val).ewm(alpha=1/period, adjust=False).mean()
    minus_di_smoothed = pd.Series(minus_dm_val).ewm(alpha=1/period, adjust=False).mean()
    
    # Handle division by zero
    df["plus_di"] = np.where(tr_smoothed > 0, 100.0 * plus_di_smoothed / tr_smoothed, 0.0)
    df["minus_di"] = np.where(tr_smoothed > 0, 100.0 * minus_di_smoothed / tr_smoothed, 0.0)
    
    di_sum = (df["plus_di"] + df["minus_di"]).abs()
    di_diff = (df["plus_di"] - df["minus_di"]).abs()
    dx = np.where(di_sum > 0, 100.0 * di_diff / di_sum, 0.0)
    
    df["adx"] = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
    return df

def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Computes MACD line, signal line, histogram, and histogram slope."""
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff().fillna(0.0)
    return df

def compute_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Average True Range (ATR) at windows of 14, 21, and 55."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Compute ATRs
    df["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    df["atr_21"] = tr.ewm(alpha=1/21, adjust=False).mean()
    df["atr_55"] = tr.ewm(alpha=1/55, adjust=False).mean()
    return df

def compute_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Computes Bollinger Bands and Band Width."""
    df["bb_mid"] = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std
    df["bb_width"] = np.where(df["bb_mid"] > 0, (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"], 0.0)
    
    # Fill NaN values in early rows
    df["bb_mid"] = df["bb_mid"].ffill().bfill()
    df["bb_upper"] = df["bb_upper"].ffill().bfill()
    df["bb_lower"] = df["bb_lower"].ffill().bfill()
    df["bb_width"] = df["bb_width"].fillna(0.0)
    return df

def compute_realized_volatility(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Computes realized volatility (rolling standard deviation of log returns)."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol"] = log_ret.rolling(window=window).std().fillna(0.0)
    return df

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Computes RSI (14) and simple divergence flags."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.nan)
    rsi = np.where(avg_loss > 0, 100.0 - (100.0 / (1.0 + rs)), 100.0)
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, rsi)
    
    df["rsi"] = rsi
    
    # RSI Divergence detection (Rolling 10-bar window)
    price_min_10 = df["close"].rolling(10).min()
    price_max_10 = df["close"].rolling(10).max()
    rsi_min_10 = df["rsi"].rolling(10).min()
    rsi_max_10 = df["rsi"].rolling(10).max()
    
    # Bullish Divergence: Price makes a new low, but RSI makes a higher low (in oversold territory < 40)
    df["rsi_bull_div"] = (df["close"] == price_min_10) & (df["rsi"] > rsi_min_10) & (df["rsi"] < 40)
    # Bearish Divergence: Price makes a new high, but RSI makes a lower high (in overbought territory > 60)
    df["rsi_bear_div"] = (df["close"] == price_max_10) & (df["rsi"] < rsi_max_10) & (df["rsi"] > 60)
    
    df["rsi_bull_div"] = df["rsi_bull_div"].astype(bool)
    df["rsi_bear_div"] = df["rsi_bear_div"].astype(bool)
    return df

def compute_session_tagging(df: pd.DataFrame) -> pd.DataFrame:
    """Tags sessions (Asian/London/NY) from UTC timestamp."""
    hours = df["timestamp"].dt.hour
    
    # Asian: 00:00 - 09:00 UTC
    df["session_asian"] = (hours >= 0) & (hours < 9)
    # London: 07:00 - 16:00 UTC
    df["session_london"] = (hours >= 7) & (hours < 16)
    # NY: 12:00 - 21:00 UTC
    df["session_ny"] = (hours >= 12) & (hours < 21)
    
    return df

def compute_swing_points(df: pd.DataFrame) -> pd.DataFrame:
    """Computes fractal-based swing highs and lows (5-bar fractal)."""
    high = df["high"]
    low = df["low"]
    
    df["swing_high"] = (
        (high > high.shift(1)) & 
        (high > high.shift(2)) & 
        (high > high.shift(-1)) & 
        (high > high.shift(-2))
    ).fillna(False)
    
    df["swing_low"] = (
        (low < low.shift(1)) & 
        (low < low.shift(2)) & 
        (low < low.shift(-1)) & 
        (low < low.shift(-2))
    ).fillna(False)
    
    return df

def compute_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes liquidity sweep / stop-hunt flag:
    Wick beyond a recent swing level (past 20 bars) followed by a close back inside.
    """
    df = compute_swing_points(df)
    
    # Extract values at swing points
    swing_high_vals = df["high"].where(df["swing_high"])
    swing_low_vals = df["low"].where(df["swing_low"])
    
    # Propagate recent swing levels (shift(1) to avoid looking at current bar's own swing if confirmed)
    recent_swing_high = swing_high_vals.shift(1).rolling(20, min_periods=1).max().ffill()
    recent_swing_low = swing_low_vals.shift(1).rolling(20, min_periods=1).min().ffill()
    
    # Swept low (bullish sweep): low went below recent swing low, but close stayed above it
    df["liquidity_sweep_bull"] = (df["low"] < recent_swing_low) & (df["close"] > recent_swing_low)
    # Swept high (bearish sweep): high went above recent swing high, but close stayed below it
    df["liquidity_sweep_bear"] = (df["high"] > recent_swing_high) & (df["close"] < recent_swing_high)
    
    df["liquidity_sweep"] = (df["liquidity_sweep_bull"] | df["liquidity_sweep_bear"]).fillna(False)
    return df

def compute_dxy_correlation(df: pd.DataFrame, dxy_df: pd.DataFrame) -> pd.DataFrame:
    """Computes rolling 20-period correlation vs DXY returns."""
    if dxy_df is None or dxy_df.empty:
        df["dxy_corr"] = 0.0
        return df
        
    # Align DXY data using merge_asof (backward match to avoid lookahead bias)
    df_sorted = df.sort_values("timestamp")
    dxy_sorted = dxy_df.sort_values("timestamp").rename(columns={"close": "dxy_close"})
    
    merged = pd.merge_asof(df_sorted, dxy_sorted[["timestamp", "dxy_close"]], on="timestamp", direction="backward")
    
    # Compute returns
    merged["gold_ret"] = merged["close"].pct_change()
    merged["dxy_ret"] = merged["dxy_close"].pct_change()
    
    # Rolling correlation
    merged["dxy_corr"] = merged["gold_ret"].rolling(20).corr(merged["dxy_ret"]).fillna(0.0)
    
    # Set back to df structure
    df["dxy_corr"] = merged["dxy_corr"].values
    return df

def compute_trend_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a trend duration counter: bars since last EMA-cross (8/21) or ADX regime change (crossing 20).
    """
    # 1. EMA cross
    ema_cross = (df["ema_8"] > df["ema_21"]) != (df["ema_8"].shift(1) > df["ema_21"].shift(1))
    
    # 2. ADX regime change (crosses 20)
    adx_cross = (df["adx"] > 20) != (df["adx"].shift(1) > 20)
    
    # Define trigger event
    is_event = (ema_cross | adx_cross).fillna(False)
    
    # Run cumsum to create groups of periods since events
    block = is_event.cumsum()
    
    # Calculate count of elements in each block
    df["trend_duration"] = df.groupby(block).cumcount()
    return df

def compute_news_proximity(df: pd.DataFrame, external_mgr=None) -> pd.DataFrame:
    """Computes the time in minutes from the bar's timestamp to the next high-impact news event."""
    if external_mgr is not None:
        # Fetch events for the next 7 days to cover the history length
        events = external_mgr.get_upcoming_events(within_minutes=10080)
        event_times = [pd.to_datetime(e["timestamp"], utc=True) for e in events if e["impact"] == "HIGH"]
        
        if event_times:
            event_times_arr = np.array(event_times)
            time_to_news = []
            
            for ts in df["timestamp"]:
                future_events = event_times_arr[event_times_arr > ts]
                if len(future_events) > 0:
                    min_future = min(future_events)
                    diff_min = (min_future - ts).total_seconds() / 60.0
                    time_to_news.append(diff_min)
                else:
                    time_to_news.append(9999.0)
            df["time_to_news"] = time_to_news
        else:
            df["time_to_news"] = 9999.0
    else:
        df["time_to_news"] = 9999.0
        
    return df

def compute_all_features(df: pd.DataFrame, dxy_df: pd.DataFrame = None, external_mgr = None) -> pd.DataFrame:
    """
    Computes all features. Operates on a copy of the input DataFrame.
    Guarantees no NaNs in the last 100 rows when df has 200+ rows.
    """
    if df.empty:
        return df
        
    # Operate on a copy
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Apply calculations
    df = compute_ema_stack(df)
    df = compute_adx(df)
    df = compute_macd(df)
    df = compute_atr(df)
    df = compute_bollinger_bands(df)
    df = compute_realized_volatility(df)
    df = compute_rsi(df)
    df = compute_session_tagging(df)
    df = compute_liquidity_sweeps(df)
    df = compute_dxy_correlation(df, dxy_df)
    df = compute_trend_duration(df)
    df = compute_news_proximity(df, external_mgr)
    
    # Final check: backfill/forward-fill any NaNs in columns to guarantee no NaNs in the final outputs
    # For early rows, EMA 200 or Bollinger standard deviations may have NaNs. This is normal,
    # but let's make sure that we fill NaNs using bfill and then ffill, except for columns like
    # swing_high/low which are boolean.
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].ffill().bfill().fillna(0.0)
        elif df[col].dtype == bool:
            df[col] = df[col].fillna(False)
            
    return df
