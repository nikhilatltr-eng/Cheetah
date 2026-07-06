import pytest
import pandas as pd
import numpy as np

def test_sell_side_cost_model_and_parity():
    # Helper mirroring sell side simulator trade execution
    def execute_sell_trade(close_entry, close_exit, spread, slippage):
        entry_price = close_entry - spread - slippage
        exit_price = close_exit + slippage
        pnl = (entry_price - exit_price) * 10.0
        return entry_price, exit_price, pnl

    # Scenario 1: A winning SELL trade (price went down)
    entry_p, exit_p, pnl = execute_sell_trade(
        close_entry=2000.0,
        close_exit=1990.0,
        spread=0.15,
        slippage=0.05
    )
    
    # Entry Bid = 2000.0 - 0.15 - 0.05 = 1999.80
    assert entry_p == 1999.80
    # Exit Ask = 1990.0 + 0.05 = 1990.05
    assert exit_p == 1990.05
    # PnL = (1999.80 - 1990.05) * 10.0 = 9.75 * 10.0 = 97.50
    assert pnl == pytest.approx(97.50)

    # Scenario 2: A losing SELL trade (price went up)
    entry_p, exit_p, pnl_loss = execute_sell_trade(
        close_entry=2000.0,
        close_exit=2010.0,
        spread=0.15,
        slippage=0.05
    )
    
    # Entry Bid = 1999.80
    # Exit Ask = 2010.05
    # PnL = (1999.80 - 2010.05) * 10.0 = -10.25 * 10.0 = -102.50
    assert pnl_loss == pytest.approx(-102.50)
