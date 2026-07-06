import os
import sys
import time
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Ensure the root folder of the project is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import main after path setup
import main

@patch("time.sleep")
@patch("telegram_alerts.requests.post")
@patch("external_data.yf.Ticker")
@patch("pandas.Timestamp.now")
@patch("yaml.safe_load")
def test_orchestrator_integration(mock_safe_load, mock_now, mock_yf_ticker, mock_tel_post, mock_sleep):
    """
    Verifies that main.py initializes successfully, runs one full cycle
    (warmup -> polling -> candle close detection -> features -> strategy -> signal evaluation),
    and exits cleanly when interrupted.
    """
    # -1. Mock config loading to supply fake MT5 and Telegram config
    real_config = {
        "symbol": "XAUUSD",
        "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"],
        "polling_interval_seconds": 5,
        "telegram": {
            "bot_token": "123456:mock_token",
            "chat_id": "987654321"
        },
        "mt5": {
            "login": "123456",
            "password": "mock_password",
            "server": "mock_server"
        }
    }
    mock_safe_load.return_value = real_config

    # 0. Setup advancing time mock for Timestamp.now
    call_count = 0
    base_time = pd.Timestamp("2026-07-05 12:00:00", tz="UTC")
    def side_effect_now(tz=None):
        nonlocal call_count
        call_count += 1
        if call_count > 8:
            return base_time + pd.Timedelta(minutes=1)
        return base_time
    mock_now.side_effect = side_effect_now

    # 1. Setup mock DXY and US10Y yields history response from yfinance
    mock_ticker_instance = MagicMock()
    mock_yf_ticker.return_value = mock_ticker_instance
    
    # Mock history df
    dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=100, freq="h", tz="UTC")
    mock_history_df = pd.DataFrame({
        "Close": [104.0] * 100
    }, index=dates)
    mock_history_df.index.name = "Datetime"
    mock_ticker_instance.history.return_value = mock_history_df
    
    # Mock Telegram request to return 200 OK
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_tel_post.return_value = mock_response
    
    # 2. Control time.sleep to run the loop exactly once
    # First sleep is polling interval; on second call, we raise KeyboardInterrupt to exit main.
    sleep_count = 0
    def side_effect(seconds):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count > 1:
            raise KeyboardInterrupt()
            
    mock_sleep.side_effect = side_effect
    
    # 3. Clean any existing integration test files or logs to have a fresh state
    if os.path.exists("cheetah.log"):
        try:
            os.remove("cheetah.log")
        except Exception:
            pass
            
    # Run the main orchestrator (forces mock mode because we are on macOS)
    # We catch KeyboardInterrupt in main, so it should exit without raising.
    main.main()
    
    # 4. Verify outputs and executions
    # Verify log file was created
    assert os.path.exists("cheetah.log")
    
    with open("cheetah.log", "r") as f:
        logs = f.read()
        
    # Assert warmup was logged
    assert "Warmup: Fetching 600 bars for XAUUSD" in logs
    assert "Warmup: Stored 600 bars for XAUUSD" in logs
    
    # Assert polling loop was entered
    assert "Entering main polling loop..." in logs
    
    # Assert a new closed candle was detected
    assert "[NEW CANDLE CLOSED]" in logs
    
    # Assert features were computed
    assert "Computing technical and macro features..." in logs
    assert "Features Computed" in logs
    
    # Assert strategy decision was made
    assert "Decision: HOLD" in logs or "Signal Generated" in logs
    
    # Assert startup message was sent to Telegram
    assert mock_tel_post.called
