import logging
import pandas as pd

logger = logging.getLogger(__name__)

def generate_signal(df: pd.DataFrame) -> dict:
    """
    Evaluates a simple rule-based strategy on the feature DataFrame.
    Rules:
      - BUY: EMA 8 crosses ABOVE EMA 21 AND ADX > 20
      - SELL: EMA 8 crosses BELOW EMA 21 AND ADX > 20
    
    Returns a dictionary structure containing direction, price, timestamp, and triggering_features.
    """
    default_signal = {
        "direction": "HOLD",
        "price": 0.0,
        "timestamp": None,
        "triggering_features": {}
    }
    
    if df is None or len(df) < 2:
        return default_signal
        
    # Get last two rows
    prev_row = df.iloc[-2]
    curr_row = df.iloc[-1]
    
    # Extract feature values
    ema_8_prev = prev_row.get("ema_8")
    ema_21_prev = prev_row.get("ema_21")
    ema_8_curr = curr_row.get("ema_8")
    ema_21_curr = curr_row.get("ema_21")
    adx_curr = curr_row.get("adx", 0.0)
    
    # Check for missing values
    if pd.isna(ema_8_prev) or pd.isna(ema_21_prev) or pd.isna(ema_8_curr) or pd.isna(ema_21_curr) or pd.isna(adx_curr):
        return default_signal
        
    # Calculate crossover
    buy_crossover = (ema_8_prev <= ema_21_prev) and (ema_8_curr > ema_21_curr)
    sell_crossover = (ema_8_prev >= ema_21_prev) and (ema_8_curr < ema_21_curr)
    
    direction = "HOLD"
    triggering_features = {}
    
    if buy_crossover and adx_curr > 20:
        direction = "BUY"
        triggering_features = {
            "ema_8_prev": float(ema_8_prev),
            "ema_21_prev": float(ema_21_prev),
            "ema_8_curr": float(ema_8_curr),
            "ema_21_curr": float(ema_21_curr),
            "adx": float(adx_curr)
        }
    elif sell_crossover and adx_curr > 20:
        direction = "SELL"
        triggering_features = {
            "ema_8_prev": float(ema_8_prev),
            "ema_21_prev": float(ema_21_prev),
            "ema_8_curr": float(ema_8_curr),
            "ema_21_curr": float(ema_21_curr),
            "adx": float(adx_curr)
        }
        
    if direction != "HOLD":
        logger.info(f"Strategy triggered {direction} signal at {curr_row['timestamp']} | Price: {curr_row['close']}")
        return {
            "direction": direction,
            "price": float(curr_row["close"]),
            "timestamp": curr_row["timestamp"],
            "triggering_features": triggering_features
        }
        
    return default_signal

def backtest_strategy(df: pd.DataFrame) -> list:
    """
    Backtests the strategy over a historical DataFrame.
    Iterates through the DataFrame and logs/returns all non-HOLD signals.
    """
    signals = []
    if df is None or len(df) < 2:
        return signals
        
    for i in range(1, len(df)):
        # Provide history up to row i
        sub_df = df.iloc[:i+1]
        sig = generate_signal(sub_df)
        if sig["direction"] != "HOLD":
            signals.append(sig)
            
    logger.info(f"Backtest complete. Generated {len(signals)} signals over {len(df)} bars.")
    return signals
