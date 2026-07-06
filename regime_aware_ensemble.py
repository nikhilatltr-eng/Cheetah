import logging
import numpy as np
import pandas as pd
from entry_model import EntryModel

logger = logging.getLogger("cheetah_ensemble")

class RegimeAwareEnsemble:
    def __init__(self):
        """
        Maintains independent entry models for each market regime state:
        trending, ranging, and volatile-news.
        """
        self.regimes_list = ["trending", "ranging", "volatile-news"]
        self.models = {regime: EntryModel() for regime in self.regimes_list}
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, regimes: np.ndarray, sample_weights: np.ndarray = None):
        """
        Splits data by regime state and trains the corresponding sub-model independently.
        """
        for regime in self.regimes_list:
            mask = (regimes == regime)
            X_sub = X[mask]
            y_sub = y[mask]
            w_sub = sample_weights[mask] if sample_weights is not None else None
            
            n_samples = len(X_sub)
            logger.info(f"Ensemble: Training sub-model for regime '{regime}' with {n_samples} samples.")
            
            if n_samples < 20:
                logger.warning(f"Ensemble: Insufficient samples ({n_samples}) for regime '{regime}'. Using base fit fallback.")
                # If too few samples, fit on the entire X dataset as fallback
                self.models[regime].fit_and_calibrate(X, y, sample_weights, X, y)
            else:
                # Split sub-samples for calibration inside sub-model
                calib_split = int(n_samples * 0.8)
                self.models[regime].fit_and_calibrate(
                    X_sub.iloc[:calib_split], y_sub[:calib_split], 
                    w_sub[:calib_split] if w_sub is not None else None,
                    X_sub.iloc[calib_split:], y_sub[calib_split:]
                )
        return self

    def predict_proba(self, X: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """Routes prediction rows to the active sub-model to compute probabilities."""
        probabilities = np.zeros((len(X), 3))
        # Default fallback class 0 (neutral)
        probabilities[:, 0] = 1.0
        
        for regime in self.regimes_list:
            mask = (regimes == regime)
            if not mask.any():
                continue
                
            X_sub = X[mask]
            sub_probs = self.models[regime].predict_proba(X_sub)
            probabilities[mask] = sub_probs
            
        return probabilities

    def predict(self, X: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """Routes prediction rows to the active sub-model to compute class predictions."""
        probs = self.predict_proba(X, regimes)
        return np.argmax(probs, axis=1)
