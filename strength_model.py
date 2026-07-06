import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)

class StrengthModel:
    def __init__(self, params: dict = None):
        """
        Move-strength estimator using LightGBM Quantile Regression.
        Fits 3 distinct models to predict p10 (lower bound), p50 (median),
        and p90 (upper bound) of forward price moves measured in ATR multiples.
        """
        self.params = params or {
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbosity": -1
        }
        self.models = {}  # Maps quantile (0.1, 0.5, 0.9) -> trained Booster/LGBMRegressor
        self.feature_cols = []

    def compute_targets(self, df: pd.DataFrame, labels_df: pd.DataFrame, atr_col: str = "atr_14") -> pd.Series:
        """
        Computes targets representing the realized trade return in ATR multiples.
        Returns: (exit_price - close_at_entry) / atr_at_entry
        """
        close = df["close"]
        exit_price = labels_df["exit_price"]
        atr = df[atr_col]
        
        # Realized return in ATR units
        move_in_atr = (exit_price - close) / (atr + 1e-6)
        # Clean any NaNs (if no exit was found or ATR is NaN)
        return move_in_atr.fillna(0.0)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: pd.Series = None):
        """Fits three separate quantile regression models on training data."""
        self.feature_cols = list(X_train.columns)
        
        quantiles = [0.1, 0.5, 0.9]
        for q in quantiles:
            logger.info(f"Training LightGBM Quantile Regressor for quantile p{int(q*100)}...")
            
            # Combine generic parameters with quantile objective
            q_params = self.params.copy()
            q_params.update({
                "objective": "quantile",
                "alpha": q
            })
            
            train_ds = lgb.Dataset(
                X_train, 
                label=y_train, 
                weight=sample_weight
            )
            
            # Train using lightgbm native booster API for flexibility
            model = lgb.train(
                q_params, 
                train_ds, 
                num_boost_round=q_params.get("n_estimators", 50)
            )
            self.models[q] = model
            
        logger.info("Quantile strength estimator training complete.")

    def predict_move_strength(self, features: pd.Series) -> dict:
        """
        Predicts move strength distribution quantiles for a single row.
        Returns:
            dict: {"p10": float, "p50": float, "p90": float, "expected_atr_multiple": float}
        """
        # Align features and reshape to 2D
        x = np.array([features[col] for col in self.feature_cols]).reshape(1, -1)
        
        predictions = {}
        for q, model in self.models.items():
            pred_val = float(model.predict(x)[0])
            predictions[f"p{int(q*100)}"] = pred_val
            
        return {
            "p10": predictions["p10"],
            "p50": predictions["p50"],
            "p90": predictions["p90"],
            "expected_atr_multiple": predictions["p50"]  # Median is the best expected value
        }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts quantiles for an entire DataFrame.
        Returns a DataFrame containing p10, p50, and p90.
        """
        X_aligned = X[self.feature_cols]
        results = {}
        for q, model in self.models.items():
            results[f"p{int(q*100)}"] = model.predict(X_aligned)
            
        res_df = pd.DataFrame(results)
        res_df["expected_atr_multiple"] = res_df["p50"]
        return res_df
