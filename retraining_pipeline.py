import os
import sys
import json
import logging
import time
import yaml
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mt5_connector import MT5Connector
from features import compute_all_features
from storage import ParquetStorage
from labeling import get_triple_barrier_labels, get_sample_weights
from validation import PurgedWalkForwardCV, compute_fold_performance
from regime_model import RegimeDetector
from entry_model import EntryModel
from strength_model import StrengthModel
from model_registry import ModelRegistry

logger = logging.getLogger("cheetah_retrain")

class RetrainingPipeline:
    def __init__(self, config_path: str = "config.yaml", registry_dir: str = "models", 
                 improvement_threshold: float = 1.05):
        """
        Retraining pipeline implementing Champion-Challenger validation
        before promoting retrained models to production.
        """
        self.config_path = config_path
        self.registry_dir = registry_dir
        self.improvement_threshold = improvement_threshold
        self.registry = ModelRegistry(registry_dir=self.registry_dir)
        
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.symbol = self.config.get("symbol", "XAUUSD")
        self.timeframe = "M1"
        self.active_versions_path = os.path.join(self.registry_dir, "active_versions.json")

    def _get_active_version(self, model_name: str) -> int:
        if os.path.exists(self.active_versions_path):
            try:
                with open(self.active_versions_path, "r") as f:
                    versions = json.load(f)
                return versions.get(model_name, 1)
            except Exception:
                pass
        return 1

    def _update_active_version(self, model_name: str, new_version: int):
        versions = {}
        if os.path.exists(self.active_versions_path):
            try:
                with open(self.active_versions_path, "r") as f:
                    versions = json.load(f)
            except Exception:
                pass
        versions[model_name] = new_version
        with open(self.active_versions_path, "w") as f:
            json.dump(versions, f, indent=4)
        logger.info(f"RetrainingPipeline: Updated active version pointer for '{model_name}' to version {new_version}")

    def run_retraining(self) -> bool:
        """
        Loads the latest history, splits it (80% fit, 20% holdout),
        trains challenger models, and compares holdout Sharpe ratio against the champion.
        Returns:
            bool: True if promoted, False if rejected.
        """
        logger.info("RetrainingPipeline: Commencing model retraining sequence...")
        
        storage = ParquetStorage(base_dir="data_store")
        df = storage.read_bars(self.symbol, self.timeframe)
        
        if len(df) < 500:
            logger.error("RetrainingPipeline: Insufficient historical data to retrain models.")
            return False
            
        # 1. Compute features & labels
        feature_df = compute_all_features(df)
        labels_df = get_triple_barrier_labels(feature_df, atr_col="atr_14")
        weights = get_sample_weights(labels_df)
        
        # Define holdout split index (last 20% chronologically)
        split_idx = int(len(feature_df) * 0.8)
        
        # 2. Evaluate current Champion on holdout window
        # Load active versions
        v_regime = self._get_active_version("regime_hmm")
        v_entry = self._get_active_version("entry_lgb")
        v_strength = self._get_active_version("strength_lgb")
        
        champ_sharpe = 0.0
        try:
            champ_regime, _ = self.registry.load_model("regime_hmm", v_regime)
            champ_entry, _ = self.registry.load_model("entry_lgb", v_entry)
            
            # Predict OOS holdout using champion
            holdout_df = feature_df.iloc[split_idx:].copy()
            holdout_labels = labels_df.iloc[split_idx:]
            
            meta_cols = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            features_cols = [c for c in feature_df.columns if c not in meta_cols]
            X_holdout = holdout_df[features_cols]
            
            champ_preds = champ_entry.predict(X_holdout)
            
            # Map predictions to trade directions [0: no-trade, 1: long, 2: short]
            # which matches validation performance expectations
            champ_signals = np.zeros(len(champ_preds))
            champ_signals[champ_preds == 1] = 1
            champ_signals[champ_preds == 2] = 2
            
            # Test indices
            test_indices = list(range(split_idx, len(feature_df)))
            champ_perf = compute_fold_performance(feature_df, champ_signals, test_indices)
            champ_sharpe = champ_perf["sharpe"]
            logger.info(f"RetrainingPipeline: Champion Model Holdout Sharpe: {champ_sharpe:.4f}")
        except Exception as e:
            logger.warning(f"RetrainingPipeline: Could not evaluate champion holdout performance: {e}. Defaulting champion Sharpe to 0.0.")
            champ_sharpe = 0.0
            
        # 3. Train Challenger models on first 80% fit window
        logger.info(f"RetrainingPipeline: Fitting challenger models on {split_idx} training samples...")
        
        fit_df = feature_df.iloc[:split_idx]
        X_fit = fit_df[features_cols]
        w_fit = weights.iloc[:split_idx].values
        
        # Prepare targets
        raw_labels = labels_df["label"].values
        y = np.zeros(len(labels_df), dtype=int)
        y[raw_labels == 1.0] = 1
        y[raw_labels == -1.0] = 2
        y_fit = y[:split_idx]
        
        # Fit Challenger Regime model
        chall_regime = RegimeDetector()
        chall_regime.fit(fit_df)
        
        # Fit Challenger Entry model
        # Use last 15% of training window for calibration
        calib_split = int(len(X_fit) * 0.85)
        chall_entry = EntryModel()
        chall_entry.fit_and_calibrate(
            X_fit.iloc[:calib_split], y_fit[:calib_split], w_fit[:calib_split],
            X_fit.iloc[calib_split:], y_fit[calib_split:]
        )
        
        # Fit Challenger Strength model
        chall_strength = StrengthModel()
        y_strength = chall_strength.compute_targets(feature_df, labels_df)
        chall_strength.fit(X_fit, y_strength.iloc[:split_idx], sample_weight=w_fit)
        
        # 4. Evaluate Challenger on holdout window
        chall_preds = chall_entry.predict(X_holdout)
        chall_signals = np.zeros(len(chall_preds))
        chall_signals[chall_preds == 1] = 1
        chall_signals[chall_preds == 2] = 2
        
        chall_perf = compute_fold_performance(feature_df, chall_signals, test_indices)
        chall_sharpe = chall_perf["sharpe"]
        logger.info(f"RetrainingPipeline: Challenger Model Holdout Sharpe: {chall_sharpe:.4f}")
        
        # 5. Champion-Challenger comparison
        # Gated promotion: Challenger must improve Sharpe ratio by self.improvement_threshold
        target_goal = champ_sharpe * self.improvement_threshold
        
        # If champion was degenerate (Sharpe <= 0), any positive Sharpe works
        promoted = False
        if (champ_sharpe <= 0 and chall_sharpe > 0) or (champ_sharpe > 0 and chall_sharpe >= target_goal):
            promoted = True
            logger.warning(
                f"RetrainingPipeline: CHALLENGER PROMOTED! Challenger Sharpe ({chall_sharpe:.4f}) "
                f"beat target threshold ({target_goal:.4f}) relative to Champion ({champ_sharpe:.4f})."
            )
            
            # Increment version numbers and save to registry
            new_v_regime = v_regime + 1
            new_v_entry = v_entry + 1
            new_v_strength = v_strength + 1
            
            dummy_metrics = {"holdout_sharpe": chall_sharpe, "retrain_time": time.time()}
            
            self.registry.save_model("regime_hmm", chall_regime, df, dummy_metrics, version=new_v_regime)
            self.registry.save_model("entry_lgb", chall_entry, df, dummy_metrics, version=new_v_entry)
            self.registry.save_model("strength_lgb", chall_strength, df, dummy_metrics, version=new_v_strength)
            
            # Update pointers
            self._update_active_version("regime_hmm", new_v_regime)
            self._update_active_version("entry_lgb", new_v_entry)
            self._update_active_version("strength_lgb", new_v_strength)
        else:
            logger.error(
                f"RetrainingPipeline: CHALLENGER REJECTED. Challenger Sharpe ({chall_sharpe:.4f}) "
                f"did not satisfy promotion threshold ({target_goal:.4f}) over Champion ({champ_sharpe:.4f})."
            )
            
        return promoted
