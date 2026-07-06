import os
import sys
import pytest
from unittest.mock import MagicMock
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_state import SharedState
from rl_exit_manager import RLExitManager

def test_rl_exit_manager_safety_enforcement():
    """
    Feeds extreme and adversarial outputs to the RLExitManager policy network
    and asserts that the outer safety shell strictly clamps and overrides decisions:
      - Trailing stop ATR multiple must be clamped within [0.5, 3.0].
      - Partial close fraction must be capped at 0.5.
      - During active risk halts, actions must be overridden to Hold.
    """
    state = SharedState(db_path="test_rl_bounds.db")
    state.clear()
    
    manager = RLExitManager(state_size=5, budget_atr_min=0.5, budget_atr_max=3.0)
    
    position = {"ticket": 123, "volume": 1.0, "partially_closed": False}
    
    # --- Scenario 1: Extreme Positive Adversarial Action [10.0, 10.0] ---
    # The policy suggests shifting trailing stop to 10 ATR and closing 100% of the trade.
    manager.evaluate_policy = MagicMock(return_value=np.array([10.0, 10.0]))
    state.set("is_halted", False)
    
    act_pos = manager.get_safe_action(
        current_pnl=150.0,
        elapsed_bars=5.0,
        atr=2.0,
        regime_idx=1.0,
        win_rate=0.55,
        position=position,
        shared_state=state
    )
    
    # Assert trailing stop is clamped to self.budget_max (3.0)
    assert act_pos["sl_adjustment"] == 3.0
    # Assert partial close is capped to exactly 0.5
    assert act_pos["partial_close_fraction"] == 0.5
    assert act_pos["action"] == "partial_close"

    # --- Scenario 2: Extreme Negative Adversarial Action [-10.0, -10.0] ---
    manager.evaluate_policy = MagicMock(return_value=np.array([-10.0, -10.0]))
    
    act_neg = manager.get_safe_action(
        current_pnl=-50.0,
        elapsed_bars=5.0,
        atr=2.0,
        regime_idx=1.0,
        win_rate=0.55,
        position=position,
        shared_state=state
    )
    
    # Assert trailing stop is clamped to self.budget_min (0.5)
    assert act_neg["sl_adjustment"] == 0.5
    # Negative output should disable partial closes
    assert act_neg["partial_close_fraction"] == 0.0
    assert act_neg["action"] == "modify_exit"

    # --- Scenario 3: System Halt Override ---
    # State flags active halt. Even with highly positive model suggestions,
    # it must force hold/null operations.
    state.set("is_halted", True)
    manager.evaluate_policy = MagicMock(return_value=np.array([1.0, 1.0]))
    
    act_halt = manager.get_safe_action(
        current_pnl=100.0,
        elapsed_bars=1.0,
        atr=2.0,
        regime_idx=1.0,
        win_rate=0.55,
        position=position,
        shared_state=state
    )
    
    assert act_halt["action"] == "hold"
    assert "Risk manager halt override" in act_halt["reason"]
    assert act_halt["sl_adjustment"] == 0.0
    assert act_halt["partial_close_fraction"] == 0.0

    # Clean up test DB
    state.clear()
    if os.path.exists("test_rl_bounds.db"):
        os.remove("test_rl_bounds.db")
