import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)

def compute_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes fast features for tick-level reversal exhaustion checks:
      - rsi: fast lookback RSI
      - adx_slope: rate of change of ADX (declining ADX indicates exhaustion)
      - wick_ratio: size of wicks relative to total range (high wick rejection)
      - vol_zscore: volume spike z-score
    """
    df = df.copy()
    
    # 1. Fast RSI (7 periods)
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(7, min_periods=1).mean()
    avg_loss = loss.rolling(7, min_periods=1).mean().replace(0.0, 1e-6)
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1.0 + rs))
    
    # 2. ADX Slope
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0.0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0.0), low_diff, 0.0)
    
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    tr_smooth = tr.rolling(7, min_periods=1).mean().replace(0.0, 1e-6)
    plus_di = 100 * pd.Series(plus_dm).rolling(7, min_periods=1).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(7, min_periods=1).mean() / tr_smooth
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, 1e-6)
    adx = dx.rolling(7, min_periods=1).mean().fillna(0.0)
    df["adx_slope"] = adx.diff().fillna(0.0)
    
    # 3. Wick rejection ratio: (high - low - abs(open - close)) / (high - low + 1e-5)
    tr_bar = df["high"] - df["low"]
    body_bar = (df["open"] - df["close"]).abs()
    df["wick_ratio"] = (tr_bar - body_bar) / (tr_bar + 1e-5)
    
    # 4. Volume z-score relative to past 30 bars
    vol_mean = df["tick_volume"].rolling(30, min_periods=1).mean()
    vol_std = df["tick_volume"].rolling(30, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)
    df["vol_zscore"] = (df["tick_volume"] - vol_mean) / vol_std
    
    return df

class ReversalModel:
    def __init__(self, armed_threshold: float = 0.6, params: dict = None):
        """
        Fast-cadence LightGBM model to detect candlestick exhaustion and wicks.
        Outputs reversal probabilities, directional bias, and arming status.
        """
        self.armed_threshold = armed_threshold
        self.params = params or {
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "verbosity": -1
        }
        self.model = None
        self.feature_cols = ["rsi", "adx_slope", "wick_ratio", "vol_zscore"]

    def compute_targets(self, df: pd.DataFrame, lookahead: int = 5) -> pd.Series:
        """
        Generates reversal labels:
          - 1: Bullish reversal (current low is local low and close rises within lookahead bars)
          - 2: Bearish reversal (current high is local high and close drops within lookahead bars)
          - 0: Neutral / No reversal
        """
        n = len(df)
        labels = np.zeros(n, dtype=int)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        # Approximate standard deviation proxy for return threshold
        rolling_std = df["close"].diff().rolling(50, min_periods=1).std().fillna(1.0).values
        
        for i in range(n - lookahead):
            std = rolling_std[i]
            # Bullish Reversal: current bar low is the minimum low of the next N bars,
            # and price rises by at least 1.5 standard deviations
            sub_lows = low[i : i + lookahead + 1]
            sub_closes = close[i : i + lookahead + 1]
            
            if low[i] == np.min(sub_lows):
                max_rise = np.max(sub_closes) - close[i]
                if max_rise >= 1.5 * std:
                    labels[i] = 1
                    continue
                    
            # Bearish Reversal: current bar high is the maximum high of the next N bars,
            # and price falls by at least 1.5 standard deviations
            sub_highs = high[i : i + lookahead + 1]
            if high[i] == np.max(sub_highs):
                max_fall = close[i] - np.min(sub_closes)
                if max_fall >= 1.5 * std:
                    labels[i] = 2
                    
        return pd.Series(labels, index=df.index)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fits the fast LightGBM reversal classifier."""
        self.feature_cols = list(X_train.columns)
        
        logger.info(f"Fitting fast Reversal Classifier on {len(X_train)} samples...")
        train_ds = lgb.Dataset(X_train, label=y_train)
        
        self.model = lgb.train(
            self.params, 
            train_ds, 
            num_boost_round=self.params.get("n_estimators", 30)
        )
        logger.info("Fast Reversal Model training complete.")

    def predict_reversal(self, fast_features: pd.Series) -> dict:
        """
        Checks for a reversal on a single feature row.
        Returns:
            dict: {"reversal_probability": float, "armed": bool, "directional_bias": int}
        """
        if self.model is None:
            return {"reversal_probability": 0.0, "armed": False, "directional_bias": 0}
            
        x = np.array([fast_features[col] for col in self.feature_cols]).reshape(1, -1)
        
        # Predict probability for [0, 1, 2] classes
        probs = self.model.predict(x)[0]
        
        prob_bullish = float(probs[1])
        prob_bearish = float(probs[2])
        
        max_prob = max(prob_bullish, prob_bearish)
        armed = max_prob > self.armed_threshold
        
        directional_bias = 0
        if armed:
            directional_bias = 1 if prob_bullish > prob_bearish else -1
            
        return {
            "reversal_probability": max_prob,
            "armed": armed,
            "directional_bias": directional_bias
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts probability distributions over classes for a DataFrame."""
        X_aligned = X[self.feature_cols]
        return self.model.predict(X_aligned)
