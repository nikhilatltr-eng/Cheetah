import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
import shap

logger = logging.getLogger(__name__)

class EntryModel:
    def __init__(self, params: dict = None):
        """
        Manages the 3-class Entry Model (LONG/SHORT/NO-TRADE) built on LightGBM.
        Includes manual Isotonic Probability Calibration and SHAP explanation capabilities.
        """
        self.params = params or {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "verbosity": -1
        }
        self.base_model = LGBMClassifier(**self.params)
        self.calibrators = {}  # Maps class_idx (0, 1, 2) -> IsotonicRegression instance
        self.feature_cols = []

    def fit_and_calibrate(self, X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray, X_calib: pd.DataFrame, y_calib: np.ndarray):
        """
        Fits the base LightGBM model on training data using uniqueness weights,
        then calibrates probabilities on a calibration set using isotonic regression for each class.
        """
        self.feature_cols = list(X_train.columns)
        
        # 1. Fit base model with sample weight
        logger.info(f"Fitting base LightGBM classifier on {len(X_train)} samples...")
        self.base_model.fit(X_train, y_train, sample_weight=sample_weight)
        
        # 2. Get raw probabilities on calibration set
        logger.info(f"Generating raw probabilities on {len(X_calib)} calibration samples...")
        raw_probs = self.base_model.predict_proba(X_calib)  # shape: (n_samples, 3)
        
        # 3. Fit independent Isotonic Regressions for each class
        logger.info("Calibrating class probabilities using Isotonic Regression...")
        self.calibrators = {}
        for c in range(3):
            # Target is 1 if class is c, else 0
            y_target = (y_calib == c).astype(float)
            
            # Predictor is the raw probability of class c
            x_input = raw_probs[:, c]
            
            # Handle edge case where calibration target lacks variance (e.g. class never occurs in calib)
            # Add dummy bounds to prevent IsotonicRegression from failing
            if len(np.unique(y_target)) < 2:
                logger.warning(f"Calibration targets for class {c} lack variance. Seeding dummy bounds.")
                x_input = np.concatenate([x_input, [0.0, 1.0]])
                y_target = np.concatenate([y_target, [0.0, 1.0]])
                
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(x_input, y_target)
            self.calibrators[c] = ir
            
        logger.info("Entry model calibration complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts calibrated probabilities. Returns array of shape (n_samples, 3)."""
        if not self.calibrators:
            raise ValueError("Model has not been trained or calibrated yet.")
            
        X_aligned = X[self.feature_cols]
        raw_probs = self.base_model.predict_proba(X_aligned)  # shape: (n_samples, 3)
        
        # Apply calibration for each class
        calib_probs = np.zeros_like(raw_probs)
        for c in range(3):
            calib_probs[:, c] = self.calibrators[c].transform(raw_probs[:, c])
            
        # Re-normalize to ensure they sum to 1.0 (probabilities)
        row_sums = calib_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        calib_probs = calib_probs / row_sums
        
        return calib_probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels: 0 (No-trade), 1 (Long), 2 (Short)."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_feature_importance(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """
        Computes SHAP values using the base tree estimator for model interpretability.
        Returns a sorted DataFrame of feature importances.
        """
        if not self.calibrators:
            raise ValueError("Model has not been trained or calibrated yet.")
            
        X_aligned = X_sample[self.feature_cols]
        
        logger.info("Calculating SHAP values for out-of-sample prediction explanation...")
        explainer = shap.TreeExplainer(self.base_model)
        shap_values = explainer(X_aligned)
        
        # Compute mean absolute SHAP value across samples
        if len(shap_values.shape) == 3:
            # Average absolute SHAP values across samples (axis 0) and classes (axis 2)
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 2))
        else:
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
            
        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": mean_abs_shap
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        
        return importance_df
