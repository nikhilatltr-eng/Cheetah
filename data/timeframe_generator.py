import os
import logging
import pandas as pd

logger = logging.getLogger("cheetah_timeframes")

class TimeframeGenerator:
    def __init__(self, processed_dir: str = "data/processed"):
        """
        Generates resampled timeframes (M5, M15, H1, H4) from cleaned M1 historical data.
        """
        self.processed_dir = processed_dir

    def aggregate_and_save(self, df_m1: pd.DataFrame, symbol: str = "XAUUSD") -> dict:
        """
        Resamples and stores candles for timeframes: M5, M15, H1, H4.
        Returns:
            dict: Summary metadata of the generated tables.
        """
        if df_m1.empty:
            logger.warning("TimeframeGenerator: Evaluated empty M1 DataFrame. No aggregations completed.")
            return {}
            
        # Ensure timestamp is a pandas DatetimeIndex for resampling
        df_work = df_m1.copy()
        df_work["timestamp"] = pd.to_datetime(df_work["timestamp"])
        df_work = df_work.set_index("timestamp")
        
        timeframe_mapping = {
            "M1": None,  # M1 is already the base input
            "M5": "5min",
            "M15": "15min",
            "H1": "1h",
            "H4": "4h"
        }
        
        summary = {}
        
        # Save clean M1 directly
        m1_dir = os.path.join(self.processed_dir, "M1")
        os.makedirs(m1_dir, exist_ok=True)
        m1_path = os.path.join(m1_dir, f"{symbol}.parquet")
        df_m1.to_parquet(m1_path)
        summary["M1"] = {"records": len(df_m1), "path": m1_path}
        
        for name, rule in timeframe_mapping.items():
            if rule is None:
                continue
                
            # Perform deterministic aggregation
            df_res = df_work.resample(rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna().reset_index()
            
            # Save to Parquet
            tf_dir = os.path.join(self.processed_dir, name)
            os.makedirs(tf_dir, exist_ok=True)
            tf_path = os.path.join(tf_dir, f"{symbol}.parquet")
            
            df_res.to_parquet(tf_path)
            summary[name] = {"records": len(df_res), "path": tf_path}
            
            logger.info(f"TimeframeGenerator: Saved {name} timeframe ({len(df_res)} candles) to {tf_path}")
            
        return summary
