import os
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from risk_manager import RiskManager

def test_daily_loss_limit_kill_switch():
    """
    Asserts that the daily loss limit halts trading once current equity
    drops below the starting daily equity minus the configured risk threshold.
    """
    # 2% Max daily loss limit
    risk_manager = RiskManager(
        max_daily_loss_pct=0.02, 
        max_positions=3,
        circuit_breaker_trades=10
    )
    
    start_equity = 10000.0
    risk_manager.reset_daily_equity_if_new_day(start_equity)
    
    # 1. Normal trading within limits (equity at $9,900 is within $200 drawdown limit)
    allowed, reason = risk_manager.evaluate_risk(
        current_equity=9900.0,
        active_positions=[],
        trade_history=[],
        proposed_direction="BUY"
    )
    assert allowed is True
    assert reason == ""
    
    # 2. Equity drops to $9,750 (drawn down by $250 which is > 2% limit of $200)
    allowed, reason = risk_manager.evaluate_risk(
        current_equity=9750.0,
        active_positions=[],
        trade_history=[],
        proposed_direction="BUY"
    )
    assert allowed is False
    assert "Daily Loss Limit hit" in reason
    assert risk_manager.is_halted is True

def test_circuit_breaker_kill_switch():
    """
    Asserts that the circuit breaker triggers and halts trading when the rolling
    win rate of the last N trades drops below the minimum performance threshold.
    """
    # Requires 35% win rate over last 10 trades
    risk_manager = RiskManager(
        max_daily_loss_pct=0.10,
        max_positions=3,
        circuit_breaker_trades=10,
        circuit_breaker_min_win_rate=0.35
    )
    
    start_equity = 10000.0
    risk_manager.reset_daily_equity_if_new_day(start_equity)
    
    # Construct a history of 10 completed trades with only 2 wins (20% win rate, violating 35% min)
    bad_history = [
        {"pnl": -50.0},
        {"pnl": -100.0},
        {"pnl": 30.0},  # Win 1
        {"pnl": -20.0},
        {"pnl": -80.0},
        {"pnl": 50.0},  # Win 2
        {"pnl": -10.0},
        {"pnl": -40.0},
        {"pnl": -150.0},
        {"pnl": -90.0}
    ]
    
    allowed, reason = risk_manager.evaluate_risk(
        current_equity=start_equity,
        active_positions=[],
        trade_history=bad_history,
        proposed_direction="BUY"
    )
    
    assert allowed is False
    assert "Performance Circuit Breaker" in reason
    assert risk_manager.is_halted is True
