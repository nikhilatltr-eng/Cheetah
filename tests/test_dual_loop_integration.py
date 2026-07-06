import os
import sys
import time
import asyncio
import datetime
import pandas as pd
import pytest
from unittest.mock import MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_state import SharedState
from slow_loop import SlowLoop
from fast_loop import FastLoop
from meta_decision_engine import MetaDecisionEngine
from reversal_model import ReversalModel

@pytest.mark.asyncio
async def test_dual_loop_concurrency():
    """
    Runs slow_loop and fast_loop concurrently under a unified async loop.
    Verifies that SharedState DB updates correctly from both writers,
    no deadlocks occur, and variables synchronize correctly.
    """
    # 1. Initialize DB state
    state = SharedState(db_path="test_dual_loop.db")
    state.clear()
    
    # 2. Setup mock connector
    connector = MagicMock()
    
    # Mock candle copy returns standard OHLCV
    now = pd.Timestamp.now(tz="UTC")
    dummy_ohlcv = pd.DataFrame({
        "timestamp": [now - pd.Timedelta(minutes=i) for i in range(100)],
        "open": [2350.0] * 100,
        "high": [2353.0] * 100,
        "low": [2348.0] * 100,
        "close": [2351.0] * 100,
        "tick_volume": [10] * 100,
        "spread": [0.2] * 100,
        "real_volume": [0] * 100
    })
    connector.fetch_ohlcv.return_value = dummy_ohlcv
    
    # Mock tick copy returns single latest tick
    connector.poll_latest_tick.return_value = {
        "timestamp": now,
        "bid": 2350.0,
        "ask": 2350.2,
        "last": 2350.0,
        "volume": 5.0
    }
    
    # Mock storage
    storage = MagicMock()
    storage.read_bars.return_value = dummy_ohlcv
    
    # 3. Setup models and engines
    meta_engine = MetaDecisionEngine()
    reversal_model = ReversalModel()
    
    # Pre-fit reversal model so predictions execute
    dummy_df = pd.DataFrame({
        "rsi": [50.0] * 30,
        "adx_slope": [0.0] * 30,
        "wick_ratio": [0.2] * 30,
        "vol_zscore": [0.0] * 30
    })
    dummy_labels = pd.Series([0] * 30)
    reversal_model.fit(dummy_df, dummy_labels)
    
    # 4. Initialize loops
    config = {
        "symbol": "XAUUSD",
        "news_events": []
    }
    
    slow = SlowLoop(config, connector, storage, state, meta_engine)
    fast = FastLoop(config, connector, state, reversal_model)
    
    # Pre-fill fast loop bars so it runs inference cycles
    for i in range(15):
        fast.fast_bars.append({
            "time": int(time.time()) - (15 - i) * 5,
            "timestamp": pd.to_datetime(int(time.time()) - (15 - i) * 5, unit="s", utc=True),
            "open": 2350.0,
            "high": 2351.0,
            "low": 2349.0,
            "close": 2350.0,
            "tick_volume": 5
        })
        
    # 5. Run both loops concurrently for a short test window
    # We run slow.run_forever and fast.run_forever with low intervals for testing speed
    slow_task = asyncio.create_task(slow.run_forever(loop_interval_seconds=0.1))
    fast_task = asyncio.create_task(fast.run_forever(interval_seconds=0.1))
    
    # Allow concurrency to execute for 1.0 second
    await asyncio.sleep(1.0)
    
    # Stop fast loop and cancel tasks
    fast.stop()
    slow_task.cancel()
    fast_task.cancel()
    
    try:
        await asyncio.gather(slow_task, fast_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass
        
    # 6. Verify shared state updates
    # Slow loop updates regime and last_updated_slow
    assert state.current_regime in ["ranging", "trending", "volatile-news"]
    assert state.last_updated_slow > 0.0
    
    # Fast loop updates reversal_probability and last_updated_fast
    assert state.get("reversal_probability") is not None
    assert state.last_updated_fast > 0.0
    
    print("\nSUCCESS: Slow loop and Fast loop executed concurrently without deadlocks.")
    
    # Clean up DB
    state.clear()
    if os.path.exists("test_dual_loop.db"):
        os.remove("test_dual_loop.db")
