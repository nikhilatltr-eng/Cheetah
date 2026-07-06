import os
import sys
import time
import logging
import datetime
import asyncio
import json
import pandas as pd

from model_registry import ModelRegistry
from shared_state import SharedState
from meta_decision_engine import MetaDecisionEngine
from features import compute_all_features

logger = logging.getLogger("cheetah_slow_loop")

class SlowLoop:
    def __init__(self, config: dict, connector, storage, shared_state: SharedState, meta_engine: MetaDecisionEngine, registry_dir: str = "models"):
        """
        Closed-candle slow loop executor.
        Runs heavy predictions (regime, entry direction, move strength) at the start of each candle,
        queries news events, and writes target trade biases to SharedState.
        """
        self.config = config
        self.connector = connector
        self.storage = storage
        self.state = shared_state
        self.meta_engine = meta_engine
        self.registry = ModelRegistry(registry_dir=registry_dir)
        
        self.symbol = config.get("symbol", "XAUUSD")
        self.timeframe = "M1"
        self.last_bar_time = None
        
        # Base models initialized to None; loaded during startup
        self.regime_model = None
        self.entry_model = None
        self.strength_model = None
        
        self.news_events = config.get("news_events", [])

    def load_models(self):
        """Loads all base ML models and the meta stacked engine if saved."""
        logger.info("SlowLoop: Loading registered machine learning models...")
        
        # Read version pointers
        v_path = os.path.join(self.registry_dir if hasattr(self, "registry_dir") else "models", "active_versions.json")
        versions = {}
        if os.path.exists(v_path):
            try:
                with open(v_path, "r") as f:
                    versions = json.load(f)
            except Exception:
                pass
                
        v_regime = versions.get("regime_hmm", 1)
        v_entry = versions.get("entry_lgb", 1)
        v_strength = versions.get("strength_lgb", 1)
        v_meta = versions.get("meta_stacker", 1)
        
        try:
            self.regime_model, regime_meta = self.registry.load_model("regime_hmm", version=v_regime)
            logger.info(f"Loaded HMM Regime Model (Version {v_regime})")
        except Exception as e:
            logger.error(f"Failed to load regime model: {e}")
            
        try:
            self.entry_model, entry_meta = self.registry.load_model("entry_lgb", version=v_entry)
            logger.info(f"Loaded Entry LightGBM Model (Version {v_entry})")
        except Exception as e:
            logger.error(f"Failed to load entry model: {e}")
            
        try:
            self.strength_model, strength_meta = self.registry.load_model("strength_lgb", version=v_strength)
            logger.info(f"Loaded Strength Model (Version {v_strength})")
        except Exception as e:
            logger.warning(f"Could not load strength model from registry: {e}. Using placeholder.")
            
        # Try loading stacked decision engine if saved
        try:
            self.meta_engine, meta_cfg = self.registry.load_model("meta_stacker", version=v_meta)
            logger.info(f"Loaded fitted Meta Decision Stacker (Version {v_meta}).")
        except Exception:
            logger.info("Meta Stacker model not found in registry. Running in heuristic/rule mode.")

    def run_once(self) -> dict:
        """Runs a single slow-loop iteration on the latest bar database."""
        logger.info("SlowLoop: Running closed-candle analysis...")
        
        # 1. Fetch latest bar sequence
        df_bars = self.connector.fetch_ohlcv(self.symbol, self.timeframe, 100)
        if df_bars.empty:
            logger.warning("SlowLoop: Failed to retrieve candles from MT5.")
            return {"action": "no_trade"}
            
        # Save to database store
        self.storage.append_bars(self.symbol, self.timeframe, df_bars)
        
        # Re-read historical bars to ensure feature indicator warmth
        df = self.storage.read_bars(self.symbol, self.timeframe)
        if len(df) < 50:
            logger.warning(f"SlowLoop: Insufficient bar count ({len(df)}) to compute features.")
            return {"action": "no_trade"}
            
        # 2. Compute features
        feature_df = compute_all_features(df)
        latest_row = feature_df.iloc[-1]
        latest_time = latest_row["timestamp"]
        
        # 3. Model predictions
        # A. Regime probabilities
        regime_probs = [0.5, 0.3, 0.2]  # Default [ranging, trending, volatile]
        regime_label = "ranging"
        if self.regime_model is not None:
            try:
                reg_res = self.regime_model.predict_regime(latest_row)
                regime_label = reg_res["state_label"]
                probs_dict = reg_res["probabilities"]
                regime_probs = [
                    probs_dict.get("ranging", 0.0),
                    probs_dict.get("trending", 0.0),
                    probs_dict.get("volatile-news", 0.0)
                ]
            except Exception as e:
                logger.error(f"Error predicting regime: {e}")
                
        # B. Entry probabilities
        entry_probs = [0.8, 0.1, 0.1]  # Default [hold, long, short]
        if self.entry_model is not None:
            try:
                # Need to convert latest row to df with correct features
                X_single = pd.DataFrame([latest_row[self.entry_model.feature_cols]])
                entry_probs = list(self.entry_model.predict_proba(X_single)[0])
            except Exception as e:
                logger.error(f"Error predicting entry signal probabilities: {e}")
                
        # C. Strength expectation
        strength_p50 = 0.5  # Default expected ATR move multiple
        if self.strength_model is not None:
            try:
                # If registered
                X_single = pd.DataFrame([latest_row[self.strength_model.feature_cols]])
                strength_p50 = float(self.strength_model.predict(X_single)["p50"].iloc[0])
            except Exception as e:
                logger.debug(f"Could not compute strength p50: {e}")
                
        # D. Reversal probability (read from SharedState, written by FastLoop)
        reversal_prob = self.state.get("reversal_probability", 0.0)
        
        # 4. Meta decision evaluation
        current_dt = datetime.datetime.now(datetime.timezone.utc)
        # Mock active positions count
        active_positions = 0 
        
        decision = self.meta_engine.decide(
            regime_probs=regime_probs,
            entry_probs=entry_probs,
            strength_p50=strength_p50,
            reversal_prob=reversal_prob,
            current_time=current_dt,
            news_events=self.news_events,
            active_positions_count=active_positions
        )
        
        # 5. Write outputs to SharedState
        self.state.current_regime = regime_label
        self.state.current_bias = decision["action"]
        self.state.last_updated_slow = time.time()
        
        logger.info(f"SlowLoop: Finished. Regime={regime_label} | Meta Bias={decision['action']}")
        return decision

    async def run_forever(self, loop_interval_seconds: float = 5.0):
        """Async loop runner checking for closed candles."""
        self.load_models()
        logger.info("SlowLoop: Orchestrator loop running in background...")
        
        while True:
            try:
                # Look at MT5 latest bar timestamp to see if a candle has closed
                df_last = self.connector.fetch_ohlcv(self.symbol, self.timeframe, 2)
                if not df_last.empty:
                    latest_bar_t = df_last.iloc[-1]["timestamp"]
                    if self.last_bar_time is None or latest_bar_t != self.last_bar_time:
                        logger.info(f"SlowLoop: New closed candle detected at {latest_bar_t}. Evaluating...")
                        self.run_once()
                        self.last_bar_time = latest_bar_t
            except Exception as e:
                logger.error(f"SlowLoop Exception in main loop: {e}")
                
            await asyncio.sleep(loop_interval_seconds)
