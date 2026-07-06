import os
import sys
import datetime
import pandas as pd
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_state import SharedState
from data_gap_detector import DataGapDetector

def test_data_gap_detection_logic():
    """
    Asserts that DataGapDetector:
      - Marks data_gap_active=False under healthy, consecutive OHLCV sequences.
      - Marks data_gap_active=True when the latest bar is older than the max_gap_seconds threshold.
      - Marks data_gap_active=True when an internal historical gap is present.
      - Automatically recovers once fresh data catchups are provided.
    """
    state = SharedState(db_path="test_gaps.db")
    state.clear()
    
    # 3-minute gap limit (180s)
    detector = DataGapDetector(
        shared_state=state,
        max_gap_seconds=180.0,
        expected_interval_seconds=60.0
    )
    
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # --- Scenario 1: Healthy sequence ---
    # Candles spaced exactly 1 minute apart, latest is 1 minute old
    times_healthy = [now - datetime.timedelta(minutes=i) for i in range(5, 0, -1)]
    df_healthy = pd.DataFrame({
        "timestamp": [t.timestamp() for t in times_healthy],
        "close": [100.0] * 5
    })
    
    has_gap = detector.check_data_gaps(df_healthy, current_time=now)
    assert has_gap is False
    assert state.get("data_gap_active") is False
    
    # --- Scenario 2: Wall-Clock Stale Delay ---
    # Latest candle is 10 minutes old
    times_stale = [now - datetime.timedelta(minutes=i) for i in range(15, 10, -1)]
    df_stale = pd.DataFrame({
        "timestamp": [t.timestamp() for t in times_stale],
        "close": [100.0] * 5
    })
    
    has_gap = detector.check_data_gaps(df_stale, current_time=now)
    assert has_gap is True
    assert state.get("data_gap_active") is True
    assert "Data Feed Stale" in state.get("data_gap_reason")
    
    # --- Scenario 3: Internal Historical Gap ---
    # Big gap in the middle of history (jump from 10m ago to 2m ago)
    times_internal = [
        now - datetime.timedelta(minutes=15),
        now - datetime.timedelta(minutes=14),
        now - datetime.timedelta(minutes=2),
        now - datetime.timedelta(minutes=1)
    ]
    df_internal = pd.DataFrame({
        "timestamp": [t.timestamp() for t in times_internal],
        "close": [100.0] * 4
    })
    
    has_gap = detector.check_data_gaps(df_internal, current_time=now)
    assert has_gap is True
    assert state.get("data_gap_active") is True
    assert "Internal Historical Data Gap" in state.get("data_gap_reason")
    
    # --- Scenario 4: Auto-Recovery ---
    # Resubmit healthy current series; should restore operational state
    has_gap = detector.check_data_gaps(df_healthy, current_time=now)
    assert has_gap is False
    assert state.get("data_gap_active") is False
    
    # Clean up test DB
    state.clear()
    if os.path.exists("test_gaps.db"):
        os.remove("test_gaps.db")
