import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class MetaDecisionEngine:
    def __init__(self, max_concurrent_positions: int = 3, news_block_minutes: int = 15):
        """
        Meta Decision Engine stacking all sub-models (Regime HMM, Entry LGBM, Strength Quantile,
        and Reversal LGBM) into a unified decision action.
        Includes physical safety overrides: position limits and news avoidance blocks.
        """
        self.max_concurrent_positions = max_concurrent_positions
        self.news_block_minutes = news_block_minutes
        self.stacker = LogisticRegression(max_iter=1000, random_state=42)
        self.is_fitted = False

    def fit(self, X_stacked: np.ndarray, y_stacked: np.ndarray):
        """
        Fits the logistic regression stacking model on base model predictions.
        X_stacked columns:
          [regime_ranging, regime_trending, regime_volatile,
           entry_hold, entry_long, entry_short,
           strength_p50, reversal_prob]
        y_stacked values:
          0: no_trade
          1: scalp_long
          2: scalp_short
          3: trend_long
          4: trend_short
        """
        logger.info(f"Training Meta Decision Stacker on {len(X_stacked)} OOS samples...")
        unique_classes = np.unique(y_stacked)
        if len(unique_classes) < 2:
            logger.warning("Insufficient target class variance to train meta engine. Falling back to default mapping.")
            self.is_fitted = False
            return
            
        self.stacker.fit(X_stacked, y_stacked)
        self.is_fitted = True
        logger.info(f"Meta Decision Stacker trained successfully. Classes: {self.stacker.classes_}")

    def decide(self, regime_probs: list, entry_probs: list, strength_p50: float, reversal_prob: float, 
               current_time: datetime.datetime, news_events: list, active_positions_count: int) -> dict:
        """
        Calculates the stacked action and enforces hard-coded rule-based overrides.
        Returns:
            dict: {"action": str, "probability": float, "reasons": list, "overridden": bool}
        """
        # Stack features into a 2D row: shape (1, 8)
        features = np.concatenate([regime_probs, entry_probs, [strength_p50, reversal_prob]]).reshape(1, -1)
        
        # 1. Base ML model prediction
        if self.is_fitted:
            try:
                pred_class = int(self.stacker.predict(features)[0])
                meta_probs = self.stacker.predict_proba(features)[0]
                pred_prob = float(meta_probs[pred_class])
            except Exception as e:
                logger.error(f"Error in meta stacker prediction: {e}. Defaulting to no_trade.")
                pred_class = 0
                pred_prob = 1.0
        else:
            # Simple heuristic mapping if the stacker was not fitted
            # Default to entry model primary direction
            # entry_probs: [hold, long, short]
            entry_direction = np.argmax(entry_probs)
            regime_idx = np.argmax(regime_probs) # 0: ranging, 1: trending
            
            if entry_direction == 1:  # Long
                pred_class = 1 if regime_idx == 0 else 3  # scalp_long vs trend_long
            elif entry_direction == 2:  # Short
                pred_class = 2 if regime_idx == 0 else 4  # scalp_short vs trend_short
            else:
                pred_class = 0
            pred_prob = float(entry_probs[entry_direction])
            
        class_mapping = {
            0: "no_trade",
            1: "scalp_long",
            2: "scalp_short",
            3: "trend_long",
            4: "trend_short"
        }
        action = class_mapping.get(pred_class, "no_trade")
        
        reasons = [
            f"Regime: Ranging={regime_probs[0]:.2f}/Trending={regime_probs[1]:.2f}/Volatile={regime_probs[2]:.2f}",
            f"Entry: Hold={entry_probs[0]:.2f}/Long={entry_probs[1]:.2f}/Short={entry_probs[2]:.2f}",
            f"Strength (p50): {strength_p50:.2f} ATR multiples",
            f"Reversal Prob: {reversal_prob:.2f}"
        ]
        
        # 2. Safety Override 1: Position Limits
        if action != "no_trade" and active_positions_count >= self.max_concurrent_positions:
            override_msg = f"Position limit reached ({active_positions_count}/{self.max_concurrent_positions})."
            logger.info(f"Override Block: {override_msg}")
            return {
                "action": "no_trade",
                "probability": 1.0,
                "reasons": reasons + [override_msg],
                "overridden": True
            }
            
        # 3. Safety Override 2: High-Impact News Proximity
        if action != "no_trade" and news_events:
            for news_time in news_events:
                if isinstance(news_time, str):
                    try:
                        n_dt = datetime.datetime.fromisoformat(news_time.replace("Z", "+00:00"))
                    except Exception:
                        continue
                else:
                    n_dt = news_time
                    
                # Calculate minutes diff
                time_diff = abs((current_time - n_dt).total_seconds()) / 60.0
                if time_diff <= self.news_block_minutes:
                    override_msg = f"News proximity block ({time_diff:.1f} mins to event at {news_time})."
                    logger.info(f"Override Block: {override_msg}")
                    return {
                        "action": "no_trade",
                        "probability": 1.0,
                        "reasons": reasons + [override_msg],
                        "overridden": True
                    }
                    
        # Traceability output
        logger.info(f"Meta Decision Engine action resolved: ACTION={action} (prob={pred_prob:.2f}) | Contributing outputs:\n  - " + "\n  - ".join(reasons))
        return {
            "action": action,
            "probability": pred_prob,
            "reasons": reasons,
            "overridden": False
        }
