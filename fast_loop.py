import os
import sys
import time
import logging
import asyncio
import datetime
import numpy as np
import pandas as pd

from shared_state import SharedState
from reversal_model import ReversalModel, compute_reversal_features
from latency_profiler import LatencyProfiler

logger = logging.getLogger("cheetah_fast_loop")

class FastLoop:
    def __init__(self, config: dict, connector, shared_state: SharedState, reversal_model: ReversalModel, 
                 median_spread: float = 0.3, spread_threshold_multiplier: float = 2.0):
        """
        High-frequency tick polling loop.
        Monitors spreads, aggregates ticks into fast candles, evaluates candlestick exhaustion
        via ReversalModel, and triggers signals if aligned with current slow-loop bias.
        """
        self.config = config
        self.connector = connector
        self.state = shared_state
        self.reversal_model = reversal_model
        
        self.symbol = config.get("symbol", "XAUUSD")
        self.median_spread = median_spread
        self.spread_multiplier = spread_threshold_multiplier
        
        # In-memory buffer to aggregate tick bars
        self.ticks = []
        self.fast_bars = []  # List of dicts representing 5-second bars
        self.max_fast_bars = 60  # Store last 5 minutes of 5s bars (5 * 12 = 60)
        
        self.last_tick_time = None
        self.is_running = False
        self.latency_profiler = LatencyProfiler(budget_ms=config.get("latency_budget_ms", 50.0))

    def check_spread_guard(self, bid: float, ask: float) -> bool:
        """
        Returns True if current spread is safe (under median_spread * multiplier).
        Returns False if spread is widened (blocking trades).
        """
        current_spread = ask - bid
        max_allowed = self.median_spread * self.spread_multiplier
        
        if current_spread > max_allowed:
            logger.warning(f"Spread Guard triggered: Current={current_spread:.2f} > MaxAllowed={max_allowed:.2f} (Widened).")
            return False
        return True

    def process_tick(self, tick: dict):
        """Aggregates a tick into fast 5-second bars."""
        tick_time = tick["timestamp"]
        price = tick["last"]
        vol = tick["volume"]
        
        # Define 5-second bucket timestamp
        # Strip timezone if datetime and convert to unix
        if isinstance(tick_time, (pd.Timestamp, datetime.datetime)):
            ts = tick_time.timestamp()
        else:
            ts = float(tick_time)
            
        bucket_ts = int(ts - (ts % 5))
        
        if not self.fast_bars or self.fast_bars[-1]["time"] != bucket_ts:
            # Create new 5s bar
            new_bar = {
                "time": bucket_ts,
                "timestamp": pd.to_datetime(bucket_ts, unit="s", utc=True),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "tick_volume": vol
            }
            self.fast_bars.append(new_bar)
            if len(self.fast_bars) > self.max_fast_bars:
                self.fast_bars.pop(0)
        else:
            # Update existing 5s bar
            bar = self.fast_bars[-1]
            bar["high"] = max(bar["high"], price)
            bar["low"] = min(bar["low"], price)
            bar["close"] = price
            bar["tick_volume"] += vol

    def run_inference_cycle(self) -> dict:
        """
        Aggregates recent bars, computes fast features, evaluates reversal,
        verifies slow-loop bias alignment, and updates SharedState.
        """
        if len(self.fast_bars) < 10:
            return {"action": "no_action", "reason": "insufficient_bars"}
            
        # Convert fast bars to DataFrame
        df_fast = pd.DataFrame(self.fast_bars)
        
        # Compute features
        df_feat = compute_reversal_features(df_fast)
        latest_row = df_feat.iloc[-1]
        
        # Predict reversal metrics
        rev_res = self.reversal_model.predict_reversal(latest_row)
        
        # Publish reversal state to SharedState
        self.state.set("reversal_probability", rev_res["reversal_probability"])
        self.state.reversal_armed = rev_res["armed"]
        self.state.last_updated_fast = time.time()
        
        # Retrieve slow loop bias context
        bias = self.state.current_bias # e.g. 'scalp_long', 'trend_long', 'no_trade'
        
        action = "no_action"
        signal_reason = "neutral"
        
        # Core alignment check: Reversal must match the slow-loop direction bias
        # Bullish reversal (directional_bias=1) matches Long bias
        # Bearish reversal (directional_bias=-1) matches Short bias
        if rev_res["armed"]:
            db = rev_res["directional_bias"]
            if db == 1 and "long" in bias:
                action = "trigger_buy"
                signal_reason = f"Bullish reversal armed matching current slow bias ({bias})"
            elif db == -1 and "short" in bias:
                action = "trigger_sell"
                signal_reason = f"Bearish reversal armed matching current slow bias ({bias})"
            else:
                signal_reason = f"Reversal armed (bias={db}) but mismatched with slow bias ({bias})"
                
        return {
            "action": action,
            "reversal_probability": rev_res["reversal_probability"],
            "armed": rev_res["armed"],
            "directional_bias": rev_res["directional_bias"],
            "slow_bias": bias,
            "reason": signal_reason
        }

    async def run_forever(self, interval_seconds: float = 1.0):
        """High-frequency async tick loop."""
        self.is_running = True
        logger.info(f"FastLoop: Polling ticks on {self.symbol} every {interval_seconds}s...")
        
        while self.is_running:
            start_time = time.perf_counter()
            try:
                tick = self.connector.poll_latest_tick(self.symbol)
                
                # Check spread guard
                bid = tick["bid"]
                ask = tick["ask"]
                spread_ok = self.check_spread_guard(bid, ask)
                
                if spread_ok:
                    # Process tick to update 5s bars
                    self.process_tick(tick)
                    
                    # Run cheap model evaluation
                    res = self.run_inference_cycle()
                    
                    if res["action"] in ["trigger_buy", "trigger_sell"]:
                        logger.info(f"⚡ FAST LOOP SIGNAL GENERATED: {res['action'].upper()} | Reason: {res['reason']}")
                        
                cycle_time_ms = (time.perf_counter() - start_time) * 1000.0
                logger.debug(f"FastLoop cycle completed in {cycle_time_ms:.2f}ms")
                self.latency_profiler.record_latency(cycle_time_ms)
                
                # Publish rolling percentiles to SharedState
                percentiles = self.latency_profiler.get_percentiles()
                self.state.set("fast_loop_p50", percentiles["p50"])
                self.state.set("fast_loop_p95", percentiles["p95"])
                self.state.set("fast_loop_p99", percentiles["p99"])
                
            except Exception as e:
                logger.error(f"FastLoop Exception: {e}")
                
            await asyncio.sleep(interval_seconds)
            
    def stop(self):
        self.is_running = False
        logger.info("FastLoop stopped.")
