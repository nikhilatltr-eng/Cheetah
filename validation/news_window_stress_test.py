import logging
import datetime
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("news_window_stress_test")

def get_major_news_events():
    """
    Returns UTC timestamps of major high-impact USD events (NFP, CPI, FOMC)
    for Q2 2026 (the OOS test period).
    """
    events = [
        # April 2026
        pd.Timestamp("2026-04-03 12:30:00", tz="UTC"),  # NFP
        pd.Timestamp("2026-04-14 12:30:00", tz="UTC"),  # CPI
        pd.Timestamp("2026-04-29 18:00:00", tz="UTC"),  # FOMC
        # May 2026
        pd.Timestamp("2026-05-01 12:30:00", tz="UTC"),  # NFP
        pd.Timestamp("2026-05-13 12:30:00", tz="UTC"),  # CPI
        # June 2026
        pd.Timestamp("2026-06-05 12:30:00", tz="UTC"),  # NFP
        pd.Timestamp("2026-06-10 12:30:00", tz="UTC"),  # CPI
        pd.Timestamp("2026-06-17 18:00:00", tz="UTC"),  # FOMC
    ]
    return events

def is_news_blackout(dt, news_events, blackout_mins=15):
    """
    Returns True if dt falls within blackout_mins minutes of any news event.
    """
    dt_aware = pd.Timestamp(dt)
    if dt_aware.tzinfo is None:
        dt_aware = dt_aware.replace(tzinfo=datetime.timezone.utc)
        
    for ev in news_events:
        diff_mins = abs((dt_aware - ev).total_seconds()) / 60.0
        if diff_mins <= blackout_mins:
            return True
    return False

def simulate_news_stress(trades_list, news_events, blackout_mins=15, blackout_multiplier=10.0):
    """
    Splits trades into blackout vs non-blackout groups.
    Applies a blowout cost multiplier to blackout trades and normal costs to non-blackout.
    Normal Cost: Spread=0.15, Slippage=0.05
    Blackout Cost: Spread=0.15*10 = 1.50, Slippage=0.05*5 = 0.25 (total friction = 1.75 points / $17.50 PnL reduction)
    """
    blackout_trades = []
    non_blackout_trades = []
    
    for t in trades_list:
        entry_time = pd.Timestamp(t["start_time"])
        in_blackout = is_news_blackout(entry_time, news_events, blackout_mins)
        
        # Recalculate PnL for blackout trades using blown-out costs
        if in_blackout:
            normal_spread = 0.15
            normal_slippage = 0.05
            normal_friction = normal_spread + 2.0 * normal_slippage  # 0.25 price points
            
            blackout_spread = normal_spread * blackout_multiplier  # 1.50
            blackout_slippage = normal_slippage * 5.0  # 0.25
            blackout_friction = blackout_spread + 2.0 * blackout_slippage  # 2.00 price points
            
            # PnL difference = (normal_friction - blackout_friction) * 10.0
            pnl_diff = (normal_friction - blackout_friction) * 10.0  # -17.50 USD reduction
            
            # Copy and modify PnL
            t_copy = dict(t)
            t_copy["pnl"] = t["pnl"] + pnl_diff
            blackout_trades.append(t_copy)
        else:
            non_blackout_trades.append(t)
            
    return blackout_trades, non_blackout_trades
