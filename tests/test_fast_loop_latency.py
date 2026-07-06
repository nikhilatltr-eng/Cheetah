import os
import sys
import time
import pandas as pd
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_state import SharedState
from reversal_model import ReversalModel
from fast_loop import FastLoop

def test_fast_loop_latency_budget():
    """
    Asserts that the fast loop inference cycle runs within the configured
    latency budget (e.g. 200ms).
    Uses mock ticks to pre-fill the candle buffer.
    """
    # 1. Initialize components with mock DB paths
    state = SharedState(db_path="test_shared_state.db")
    state.clear()
    
    reversal_model = ReversalModel()
    
    # Fit model on dummy data to ensure prediction works
    dummy_df = pd.DataFrame({
        "rsi": [50.0] * 30,
        "adx_slope": [0.0] * 30,
        "wick_ratio": [0.2] * 30,
        "vol_zscore": [0.0] * 30
    })
    dummy_labels = pd.Series([0] * 30)
    reversal_model.fit(dummy_df, dummy_labels)
    
    # 2. Setup loop
    config = {"symbol": "XAUUSD"}
    fast_loop = FastLoop(
        config=config,
        connector=None,
        shared_state=state,
        reversal_model=reversal_model
    )
    
    # Pre-fill with 15 fast bars of mock data
    now_ts = int(time.time())
    for i in range(15):
        fast_loop.fast_bars.append({
            "time": now_ts - (15 - i) * 5,
            "timestamp": pd.to_datetime(now_ts - (15 - i) * 5, unit="s", utc=True),
            "open": 2350.0,
            "high": 2352.0,
            "low": 2349.0,
            "close": 2351.0,
            "tick_volume": 5
        })
        
    # 3. Benchmark a single inference cycle
    start_time = time.perf_counter()
    res = fast_loop.run_inference_cycle()
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000.0
    latency_budget_ms = 200.0
    
    print(f"\nFast Loop Inference latency: {latency_ms:.4f} ms (Budget: {latency_budget_ms} ms)")
    
    # Assert cycle runs well below the 200ms latency budget
    assert latency_ms < latency_budget_ms, f"Latency budget exceeded: {latency_ms:.2f}ms > {latency_budget_ms}ms"
    
    # Assert state variables were correctly written
    assert state.get("reversal_probability") is not None
    assert state.reversal_armed is not None
    
    # Clean up DB
    state.clear()
    if os.path.exists("test_shared_state.db"):
        os.remove("test_shared_state.db")
