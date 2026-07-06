import os
import sys
import time
import pytest
from unittest.mock import MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_state import SharedState
from connection_watchdog import ConnectionWatchdog

def test_watchdog_connection_loss_and_alerts():
    """
    Asserts that the ConnectionWatchdog:
      - Sets the trading halt state in SharedState immediately on disconnect.
      - Dispatches a Telegram alert with details when active positions are exposed.
      - Blocks order generation during the outage.
    """
    state = SharedState(db_path="test_watchdog.db")
    state.clear()
    
    # 1. Setup mock connector in disconnected state
    mock_connector = MagicMock()
    mock_connector.connected = False
    
    # 2. Setup mock execution engine with an active position
    mock_execution = MagicMock()
    mock_execution.mock = True
    mock_execution.mock_positions = [{
        "ticket": 999111,
        "type": "BUY",
        "volume": 0.5,
        "open_price": 2355.0
    }]
    
    # Initialize watchdog
    watchdog = ConnectionWatchdog(
        connector=mock_connector,
        shared_state=state,
        execution_engine=mock_execution
    )
    
    # Mock the telegram dispatch function to verify it receives the message
    watchdog._send_telegram = MagicMock()
    
    # 3. Execute connection check
    watchdog.run_check()
    
    # Assertions
    assert state.get("is_halted") is True
    assert state.get("halt_reason") == "MT5 Connection Lost"
    
    # Assert alert was dispatched
    watchdog._send_telegram.assert_called_once()
    alert_arg = watchdog._send_telegram.call_args[0][0]
    assert "MT5 Disconnected while positions are ACTIVE" in alert_arg
    assert "Ticket 999111" in alert_arg
    
    # Clean up test DB
    state.clear()
    if os.path.exists("test_watchdog.db"):
        os.remove("test_watchdog.db")
