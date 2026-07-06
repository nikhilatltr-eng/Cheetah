import logging
import pandas as pd
import numpy as np
import datetime

logger = logging.getLogger("cheetah_cleaner")

class DataCleaner:
    def __init__(self, max_spike_pct: float = 0.02):
        """
        Cleans and sanitizes historical candle data.
        Fills gaps using forward-fill (no fabricated price interpolation) and resolves spikes.
        """
        self.max_spike_pct = max_spike_pct

    def clean_m1_data(self, df: pd.DataFrame) -> tuple:
        """
        Runs the full cleaning pipeline on M1 data.
        Returns:
            (pd.DataFrame, dict): The cleaned DataFrame and a metrics report dictionary.
        """
        if df.empty:
            return df, {"status": "empty"}
            
        initial_len = len(df)
        df = df.copy()
        if "volume" not in df.columns and "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        if "tick_volume" not in df.columns and "volume" in df.columns:
            df["tick_volume"] = df["volume"]
        if "volume" not in df.columns:
            df["volume"] = 0.0
            df["tick_volume"] = 0.0
        
        # 1. De-duplicate timestamps
        df = df.drop_duplicates(subset=["timestamp"])
        dup_count = initial_len - len(df)
        
        # 2. Sort chronologically
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        
        # 3. Remove negative or zero prices
        valid_prices_mask = (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
        df = df[valid_prices_mask].copy()
        invalid_count = len(valid_prices_mask) - df["open"].count()
        
        # 4. Correct internal OHLC inconsistencies (e.g. rounding errors: High < Close)
        # We repair them to be physically consistent instead of throwing away the row
        df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
        df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))
        
        # 5. Handle missing periods (reindex to a complete 1-minute grid)
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        
        start_t = df.index.min()
        end_t = df.index.max()
        
        # Reindex to full minute grid
        full_grid = pd.date_range(start=start_t, end=end_t, freq="1min", tz=datetime.timezone.utc if start_t.tzinfo else None)
        df = df.reindex(full_grid)
        
        missing_count = df["close"].isna().sum()
        
        # Forward fill prices (Open, High, Low, Close get last Close price)
        df["close"] = df["close"].ffill()
        df["open"] = df["open"].fillna(df["close"])
        df["high"] = df["high"].fillna(df["close"])
        df["low"] = df["low"].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0.0)
        
        df.index.name = "timestamp"
        df = df.reset_index()
        
        # 6. Detect and correct abnormal price spikes
        # If price shifts > max_spike_pct on M1, we cap/correct it using adjacent value
        spikes_detected = 0
        closes = df["close"].values.copy()
        opens = df["open"].values.copy()
        highs = df["high"].values.copy()
        lows = df["low"].values.copy()
        
        for i in range(1, len(df)):
            pct_change = abs(closes[i] - closes[i - 1]) / closes[i - 1]
            if pct_change > self.max_spike_pct:
                spikes_detected += 1
                # Correct spike: reset candle to previous Close
                closes[i] = closes[i - 1]
                opens[i] = closes[i - 1]
                highs[i] = closes[i - 1]
                lows[i] = closes[i - 1]
                
        df["open"] = opens
        df["high"] = highs
        df["low"] = lows
        df["close"] = closes
        
        report = {
            "initial_records": int(initial_len),
            "final_records": int(len(df)),
            "duplicates_removed": int(dup_count),
            "invalid_price_records_removed": int(invalid_count),
            "missing_bars_filled": int(missing_count),
            "abnormal_spikes_corrected": int(spikes_detected),
            "start_time": start_t.isoformat() if hasattr(start_t, "isoformat") else str(start_t),
            "end_time": end_t.isoformat() if hasattr(end_t, "isoformat") else str(end_t),
            "data_quality_pct": float(1.0 - (missing_count / len(df))) if len(df) > 0 else 0.0
        }
        
        logger.info(f"DataCleaner: Complete. Final records: {len(df)} | Quality: {report['data_quality_pct']:.2%}")
        return df, report
