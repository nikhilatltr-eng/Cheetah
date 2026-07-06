import os
import sys
import pandas as pd
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_state import SharedState
from drift_monitor import PerformanceDriftMonitor

def test_drift_detection_noise_vs_signal():
    """
    Asserts that PerformanceDriftMonitor:
      - Does NOT flag normal noise/variance (e.g., win rate around 52% baseline).
      - Correctly flags structural drift (e.g., win rate drops 20 points to 30%).
    """
    state = SharedState(db_path="test_drift.db")
    state.clear()
    
    # Initialize monitor expecting 52% win rate
    monitor = PerformanceDriftMonitor(
        shared_state=state,
        expected_win_rate=0.52,
        expected_r_mean=0.20,
        expected_r_std=1.0,
        min_trades_to_test=30,
        p_value_alpha=0.05
    )
    
    # --- Scenario 1: Normal Variance ---
    # Generate 50 trades with a 54% win rate (very close to expected 52%)
    np.random.seed(42)
    normal_pnl = [50.0 if x else -45.0 for x in (np.random.rand(50) < 0.54)]
    
    normal_df = pd.DataFrame({
        "ticket": range(1, 51),
        "pnl": normal_pnl,
        # R-multiples close to normal expectations
        "r_multiple": np.random.normal(0.20, 1.0, 50)
    })
    
    res_normal = monitor.evaluate_drift(normal_df)
    
    # Assert normal variance is NOT flagged as drift
    assert res_normal["drift_detected"] is False
    assert state.get("is_halted", False) is False
    
    # --- Scenario 2: Injected Performance Drift ---
    # Win rate drops to 30% (severe structural degradation)
    drift_pnl = [50.0 if x else -50.0 for x in (np.random.rand(50) < 0.30)]
    # R-multiples drop to a negative expectation
    drift_r = np.random.normal(-0.50, 1.0, 50)
    
    drift_df = pd.DataFrame({
        "ticket": range(1, 51),
        "pnl": drift_pnl,
        "r_multiple": drift_r
    })
    
    res_drift = monitor.evaluate_drift(drift_df)
    
    # Assert structural drift IS flagged
    assert res_drift["drift_detected"] is True
    # Assert the circuit breaker was engaged in SharedState
    assert state.get("is_halted") is True
    assert "Win Rate degradation" in state.get("halt_reason") or "R-multiple distribution shift" in state.get("halt_reason")
    
    # Clean up test DB
    state.clear()
    if os.path.exists("test_drift.db"):
        os.remove("test_drift.db")
