import os
import sys
import time
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_state import SharedState
from capital_scaler import CapitalScaler

def test_capital_scaler_stage_advancement_and_reversion():
    """
    Asserts that the CapitalScaler:
      - Blocks stage advancement when trade count gates are not met.
      - Advances stage when all gates (trades, days, drawdown) are satisfied.
      - Reverts stage when drawdown exceeds the configured reversion threshold (4%).
    """
    state = SharedState(db_path="test_scaler.db")
    state.clear()
    
    # 10 trades, 5 days, 4% drawdown limit
    scaler = CapitalScaler(
        shared_state=state,
        min_trades_per_stage=10,
        min_days_per_stage=5,
        max_drawdown_limit=0.04
    )
    
    # Assert initial stage
    assert scaler.current_stage == 1
    
    # --- Scenario 1: Gates Not Met (Incomplete Trades) ---
    # Only 5 trades completed (requires 10)
    scaler.process_closed_trades(
        current_equity=10000.0,
        trades_completed_at_stage=[{"pnl": 10.0}] * 5
    )
    # Stage remains 1
    assert state.get("scaler_stage") == 1
    
    # --- Scenario 2: Advancement Gates Met ---
    # Manually shift start time back 6 days to satisfy duration gate
    six_days_ago = time.time() - (6 * 86400)
    state.set("scaler_stage_start_time", six_days_ago)
    
    # Process with 12 trades
    scaler.process_closed_trades(
        current_equity=10050.0,
        trades_completed_at_stage=[{"pnl": 10.0}] * 12
    )
    # Stage advances to 2
    assert state.get("scaler_stage") == 2
    
    # --- Scenario 3: Drawdown Reversion ---
    # Current stage is 2. Let's record peak equity of Stage 2 as $10,100
    state.set("scaler_stage_peak_equity", 10100.0)
    
    # Equity drops to $9,650 (drawdown = (10100 - 9650)/10100 = 4.45% which is >= 4% limit)
    scaler.process_closed_trades(
        current_equity=9650.0,
        trades_completed_at_stage=[]
    )
    
    # Scaler reverts back to Stage 1!
    assert state.get("scaler_stage") == 1
    
    # Clean up test DB
    state.clear()
    if os.path.exists("test_scaler.db"):
        os.remove("test_scaler.db")
