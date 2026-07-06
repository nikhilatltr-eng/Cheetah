import logging
import datetime
import pandas as pd
import numpy as np
from shared_state import SharedState

logger = logging.getLogger("cheetah_gap_detector")

class DataGapDetector:
    def __init__(self, shared_state: SharedState, max_gap_seconds: float = 180.0, 
                 expected_interval_seconds: float = 60.0):
        """
        Scans OHLCV bars for timestamp gaps that could cause features to compute on stale or incomplete data.
        """
        self.state = shared_state
        self.max_gap_seconds = max_gap_seconds
        self.expected_interval = expected_interval_seconds

    def check_data_gaps(self, df_bars: pd.DataFrame, current_time: datetime.datetime = None) -> bool:
        """
        Scans bars for internal gaps and evaluates latency of the latest candle relative to the wall clock.
        Sets 'data_gap_active' to True/False in SharedState.
        Returns:
            bool: True if a gap is active (data is stale/broken), False if data is healthy.
        """
        if df_bars.empty:
            logger.warning("DataGapDetector: Evaluated empty candle sequence. Flagging data gap.")
            self.state.set("data_gap_active", True)
            return True
            
        # Ensure current_time is timezone-aware
        if current_time is None:
            current_time = datetime.datetime.now(datetime.timezone.utc)
        elif current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=datetime.timezone.utc)
            
        # Parse latest bar time
        latest_bar_time = df_bars["timestamp"].iloc[-1]
        if isinstance(latest_bar_time, (int, float, np.integer, np.floating)):
            # convert from epoch seconds
            latest_bar_dt = datetime.datetime.fromtimestamp(latest_bar_time, datetime.timezone.utc)
        else:
            latest_bar_dt = pd.to_datetime(latest_bar_time)
            if latest_bar_dt.tzinfo is None:
                latest_bar_dt = latest_bar_dt.replace(tzinfo=datetime.timezone.utc)
                
        # 1. Evaluate Wall-Clock Delay (is the latest bar stale?)
        wall_clock_delay = (current_time - latest_bar_dt).total_seconds()
        
        # 2. Evaluate Internal Sequence Gaps (are there holes in history?)
        # Calculate diffs between consecutive bars
        max_internal_gap = 0.0
        if len(df_bars) > 1:
            timestamps = pd.to_datetime(df_bars["timestamp"])
            diffs = timestamps.diff().dropna().dt.total_seconds()
            if not diffs.empty:
                max_internal_gap = float(diffs.max())
            
        gap_detected = False
        reason = ""
        
        if wall_clock_delay >= self.max_gap_seconds:
            gap_detected = True
            reason = f"Data Feed Stale: Latest bar timestamp {latest_bar_dt.isoformat()} is {wall_clock_delay:.1f}s behind current time {current_time.isoformat()} (Limit: {self.max_gap_seconds}s)."
        elif max_internal_gap > (self.expected_interval * 2.5):  # allow a buffer of 2.5x expected interval
            gap_detected = True
            reason = f"Internal Historical Data Gap detected: Max gap is {max_internal_gap:.1f}s (Expected: {self.expected_interval}s)."
            
        if gap_detected:
            logger.error(f"DataGapDetector: {reason}. Pausing entry signals.")
            self.state.set("data_gap_active", True)
            self.state.set("data_gap_reason", reason)
        else:
            if self.state.get("data_gap_active", False):
                logger.info("DataGapDetector: Data feeds synchronized. Resuming signal generation.")
            self.state.set("data_gap_active", False)
            self.state.set("data_gap_reason", "")
            
        return gap_detected
