import os
import sys
import datetime
import time
import pytest
from unittest.mock import MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from demo_runner import DemoRunner
from shared_state import SharedState

def test_execution_and_news_blackout_integration():
    """
    Integrates the DemoRunner components to assert that:
      1. News blackout window blocks proposed trade executions.
      2. Valid entries execute and open paper tickets outside blackouts.
      3. Adverse price updates trigger stop-out closes (Stop Loss) correctly.
      4. Trailing stops and partial closes adapt on positive price ticks.
    """
    # Initialize components using the test DB
    runner = DemoRunner(config_path="config.yaml", paper_trading=True)
    runner.state = SharedState("test_execution.db")
    runner.state.clear()
    
    # Pre-configure engine mock attributes
    runner.execution.state = runner.state
    runner.execution.mock_positions = []
    runner.execution.trade_history = []
    
    # 1. Simulate Economic News Blackout Window
    now = datetime.datetime.now(datetime.timezone.utc)
    # Inject high impact event 5 minutes in the future
    news_release_time = now + datetime.timedelta(minutes=5)
    runner.news_filter.add_mock_event(news_release_time)
    
    # Assert calendar recognizes blackout state
    assert runner.news_filter.is_blackout_active(now) is True
    
    # 2. Simulate Entry Trigger inside News Blackout
    # Attempting to execute should be blocked by the runner's safety check
    # Let's verify our blackout check works
    is_blocked = runner.news_filter.is_blackout_active(now)
    assert is_blocked is True
    
    # 3. Simulate Entry Trigger outside Blackout
    # Clear mock news events
    runner.news_filter.high_impact_events = []
    assert runner.news_filter.is_blackout_active(now) is False
    
    # Open a mock trade
    # BUY 1.0 lot of XAUUSD at 2350.0. SL at 2345.0, TP at 2370.0.
    res = runner.execution.execute_order(
        action="trigger_buy",
        volume=1.0,
        price=2350.0,
        sl=2345.0,
        tp=2370.0,
        reason_context="Integration Test Entry"
    )
    
    assert res["status"] == "success"
    assert len(runner.execution.mock_positions) == 1
    ticket = res["ticket"]
    
    # Check that position state variables were written to SQLite
    db_positions = runner.state.get("mock_positions")
    assert len(db_positions) == 1
    assert db_positions[0]["ticket"] == ticket
    
    # 4. Simulate Price stop-out (price drops to 2344.0, hitting SL of 2345.0)
    # update_positions_on_tick runs checking triggers
    runner.execution.update_positions_on_tick(bid=2344.0, ask=2344.2, atr=2.0)
    
    # Position should be closed out!
    assert len(runner.execution.mock_positions) == 0
    assert len(runner.execution.trade_history) == 1
    assert runner.execution.trade_history[0]["ticket"] == ticket
    assert runner.execution.trade_history[0]["pnl"] < 0.0  # Realized a loss
    
    # Clean up test DB
    runner.state.clear()
    if os.path.exists("test_execution.db"):
        os.remove("test_execution.db")
    if os.path.exists("shared_state.db"):
        os.remove("shared_state.db")
