import logging
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import lightgbm as lgb

logger = logging.getLogger(__name__)

class RegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.hmm = GaussianHMM(
            n_components=n_regimes, 
            covariance_type="diag", 
            n_iter=100, 
            random_state=42
        )
        self.cross_check_model = None
        self.state_map = {}  # Maps index -> "ranging", "trending", "volatile-news"
        self.feature_cols = ["realized_vol", "adx", "trend_duration"]

    def fit(self, df: pd.DataFrame):
        """Fits the Gaussian HMM on historical features and maps hidden states to labels."""
        X = df[self.feature_cols].copy()
        # Clean NaNs
        X = X.dropna()
        
        if len(X) < 50:
            raise ValueError(f"Insufficient historical data to fit HMM (Length: {len(X)}).")
            
        logger.info(f"Fitting Gaussian HMM with {self.n_regimes} states on {len(X)} rows...")
        self.hmm.fit(X)
        
        # Map HMM states to regime labels based on feature means
        means = self.hmm.means_  # shape: (n_components, n_features)
        # Features index mapping: 0 -> realized_vol, 1 -> adx, 2 -> trend_duration
        
        vol_means = means[:, 0]
        adx_means = means[:, 1]
        
        # 1. State with the highest volatility mean is mapped to 'volatile-news'
        volatile_news_state = np.argmax(vol_means)
        
        # 2. Of the remaining two states, the one with the higher ADX mean is 'trending',
        # while the other is 'ranging'.
        remaining_states = [i for i in range(self.n_regimes) if i != volatile_news_state]
        
        if adx_means[remaining_states[0]] > adx_means[remaining_states[1]]:
            trending_state = remaining_states[0]
            ranging_state = remaining_states[1]
        else:
            trending_state = remaining_states[1]
            ranging_state = remaining_states[0]
            
        self.state_map = {
            ranging_state: "ranging",
            trending_state: "trending",
            volatile_news_state: "volatile-news"
        }
        
        logger.info(f"HMM state mappings established: {self.state_map}")
        logger.info(f"HMM state parameter means (vol, adx, duration):\n{means}")
        
        # Fit rule-based classifier cross-check
        self._fit_cross_check(df)

    def _fit_cross_check(self, df: pd.DataFrame):
        """Fits a LightGBM sanity crosscheck model on rule-based labeled windows."""
        df = df.copy()
        
        # Rule 1: Volatility Z-score > 2
        vol = df["realized_vol"]
        vol_mean = vol.rolling(100, min_periods=1).mean()
        vol_std = vol.rolling(100, min_periods=1).std().fillna(0.0001)
        vol_z = (vol - vol_mean) / vol_std
        is_volatile = vol_z > 2.0
        
        # Rule 2: Sustained ADX > 25 (minimum over 5 bars > 25)
        is_trending = df["adx"].rolling(5, min_periods=1).min() > 25.0
        
        # Rule 3: ADX < 20
        is_ranging = df["adx"] < 20.0
        
        # Encode default ranging (0), trending (1), volatile (2)
        labels = np.zeros(len(df), dtype=int)
        labels[is_trending] = 1
        labels[is_volatile] = 2
        
        X_train = df[self.feature_cols].copy()
        valid = X_train.notna().all(axis=1)
        
        X_clean = X_train[valid]
        y_clean = labels[valid]
        
        if len(X_clean) > 20:
            train_ds = lgb.Dataset(X_clean, label=y_clean)
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "verbosity": -1,
                "seed": 42
            }
            self.cross_check_model = lgb.train(params, train_ds, num_boost_round=30)
            logger.info("Regime crosscheck LightGBM classifier successfully trained.")
        else:
            logger.warning("Insufficient data to train rule-based crosscheck model.")

    def predict_regime(self, feature_row: pd.Series) -> dict:
        """
        Predicts the current market regime for a feature row.
        Returns:
            dict: {"state_label": str, "probabilities": dict}
        """
        # Align features
        x = np.array([feature_row[col] for col in self.feature_cols]).reshape(1, -1)
        
        # 1. HMM Prediction
        state_idx = int(self.hmm.predict(x)[0])
        state_label = self.state_map.get(state_idx, "unknown")
        
        # State Probabilities
        probs = self.hmm.predict_proba(x)[0]
        probabilities = {self.state_map.get(i, f"state_{i}"): float(probs[i]) for i in range(self.n_regimes)}
        
        # 2. Crosscheck Prediction
        crosscheck_label = "ranging"
        if self.cross_check_model is not None:
            pred_cc = self.cross_check_model.predict(x)[0]
            cc_idx = int(np.argmax(pred_cc))
            cc_map = {0: "ranging", 1: "trending", 2: "volatile-news"}
            crosscheck_label = cc_map.get(cc_idx, "ranging")
            
        return {
            "state_label": state_label,
            "probabilities": probabilities,
            "crosscheck_label": crosscheck_label
        }
