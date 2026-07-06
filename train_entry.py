import os
import sys
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mt5_connector import MT5Connector
from features import compute_all_features
from storage import ParquetStorage
from labeling import get_triple_barrier_labels, get_sample_weights
from validation import PurgedWalkForwardCV, compute_fold_performance, generate_fold_report

# Phase 2/3 Models
from regime_model import RegimeDetector
from entry_model import EntryModel
from strength_model import StrengthModel
from reversal_model import ReversalModel, compute_reversal_features
from meta_decision_engine import MetaDecisionEngine
from model_registry import ModelRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_pipeline")

def map_meta_class(df_row, label_val, regime_label) -> int:
    """
    Maps outcomes to 5 meta decision classes:
      0: no_trade
      1: scalp_long (profit hit in ranging)
      2: scalp_short (loss hit / short profit in ranging)
      3: trend_long (profit hit in trending)
      4: trend_short (loss hit / short profit in trending)
    """
    if label_val == 0:
        return 0
    elif label_val == 1:
        return 1 if regime_label == "ranging" else 3
    elif label_val == -1:
        return 2 if regime_label == "ranging" else 4
    return 0

def main():
    logger.info("Starting Cheetah full pipeline model training (Phase 3)...")
    
    # 1. Load configuration settings
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    label_cfg = config.get("labeling", {"pt_mult": 2.0, "sl_mult": 2.0, "max_holding_bars": 20})
    val_cfg = config.get("validation", {"n_folds": 5, "embargo_bars": 10})
    
    pt_mult = label_cfg.get("pt_mult", 2.0)
    sl_mult = label_cfg.get("sl_mult", 2.0)
    max_holding_bars = label_cfg.get("max_holding_bars", 20)
    
    n_folds = val_cfg.get("n_folds", 5)
    embargo_bars = val_cfg.get("embargo_bars", 10)
    
    symbol = config.get("symbol", "XAUUSD")
    timeframe = "M1"
    
    # 2. Retrieve history
    storage = ParquetStorage(base_dir="data_store")
    connector = MT5Connector(mock=True)
    
    df = storage.read_bars(symbol, timeframe)
    if df.empty or len(df) < 500:
        logger.info("Storage is empty or insufficient. Fetching warmup mock bars to seed storage...")
        df_warmup = connector.fetch_ohlcv(symbol, timeframe, 3000)
        storage.append_bars(symbol, timeframe, df_warmup)
        df = storage.read_bars(symbol, timeframe)
        
    logger.info(f"Loaded {len(df)} historical bars for training.")
    
    # 3. Compute features
    logger.info("Computing features...")
    feature_df = compute_all_features(df)
    
    # 4. Generate labels & weights
    logger.info("Generating triple-barrier labels...")
    labels_df = get_triple_barrier_labels(
        feature_df, 
        atr_col="atr_14", 
        pt_mult=pt_mult, 
        sl_mult=sl_mult, 
        max_holding_bars=max_holding_bars
    )
    
    # Compute uniqueness weights
    logger.info("Calculating sample uniqueness weights...")
    weights = get_sample_weights(labels_df)
    
    # Target label: map entry model labels to classes: 0 (no-trade), 1 (long), 2 (short)
    raw_labels = labels_df["label"].values
    y = np.zeros(len(labels_df), dtype=int)
    y[raw_labels == 1.0] = 1
    y[raw_labels == -1.0] = 2
    
    # Prepare features list
    meta_cols = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    features_cols = [c for c in feature_df.columns if c not in meta_cols]
    X = feature_df[features_cols].copy()
    
    # 5. Prepare walk-forward cross validation
    cv = PurgedWalkForwardCV(n_folds=n_folds, embargo_bars=embargo_bars)
    
    # Stacking inputs container
    stacked_X_list = []
    stacked_y_list = []
    
    # Hold prediction arrays for performance comparisons
    all_oos_y_true = []
    all_oos_entry_preds = []
    all_oos_meta_preds = []
    all_oos_entry_probs = []
    
    logger.info("Executing purged walk-forward base model validation splits...")
    fold_idx = 1
    for train_idx, test_idx in cv.split(feature_df, labels_df):
        if len(train_idx) < 50:
            continue
            
        logger.info(f"--- FOLD {fold_idx} ---")
        
        # Partition training set into fit and calibration (80/20 chronological split)
        split_point = int(len(train_idx) * 0.8)
        fit_idx = train_idx[:split_point]
        calib_idx = train_idx[split_point:]
        
        X_fit, y_fit = X.iloc[fit_idx], y[fit_idx]
        w_fit = weights.iloc[fit_idx].values
        
        X_calib, y_calib = X.iloc[calib_idx], y[calib_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        
        # Fit Base models inside the loop to ensure out-of-sample predictions
        # A. Regime Model HMM
        fold_regime = RegimeDetector()
        fold_regime.fit(feature_df.iloc[fit_idx])
        
        # B. Entry Classifier
        fold_entry = EntryModel()
        fold_entry.fit_and_calibrate(X_fit, y_fit, w_fit, X_calib, y_calib)
        
        # C. Strength Quantile Regressors
        fold_strength = StrengthModel()
        y_strength_target = fold_strength.compute_targets(feature_df, labels_df)
        fold_strength.fit(X_fit, y_strength_target.iloc[fit_idx], sample_weight=w_fit)
        
        # D. Reversal Model Classifier
        fold_reversal = ReversalModel()
        reversal_feat_df = compute_reversal_features(feature_df)
        reversal_features_cols = ["rsi", "adx_slope", "wick_ratio", "vol_zscore"]
        y_reversal_target = fold_reversal.compute_targets(reversal_feat_df)
        fold_reversal.fit(
            reversal_feat_df.iloc[fit_idx][reversal_features_cols],
            y_reversal_target.iloc[fit_idx]
        )
        
        # Predict out-of-sample on test set
        # A. Entry probabilities
        entry_probs_oos = fold_entry.predict_proba(X_test)
        entry_preds_oos = fold_entry.predict(X_test)
        
        # B. Regime probabilities
        regime_probs_list = []
        regime_labels_list = []
        for _, row in feature_df.iloc[test_idx].iterrows():
            reg_res = fold_regime.predict_regime(row)
            probs_dict = reg_res["probabilities"]
            regime_probs_list.append([
                probs_dict.get("ranging", 0.0),
                probs_dict.get("trending", 0.0),
                probs_dict.get("volatile-news", 0.0)
            ])
            regime_labels_list.append(reg_res["state_label"])
        regime_probs_oos = np.array(regime_probs_list)
        
        # C. Strength p50 predictions
        strength_preds_df = fold_strength.predict(X_test)
        strength_p50_oos = strength_preds_df["p50"].values
        
        # D. Reversal probabilities
        reversal_probs_all = fold_reversal.predict(reversal_feat_df.iloc[test_idx][reversal_features_cols])
        # Reversal prob is max probability of reversal classes (class 1 or class 2)
        reversal_prob_oos = np.max(reversal_probs_all[:, 1:3], axis=1)
        
        # Map stacked 5-class target: [0: no_trade, 1: scalp_long, 2: scalp_short, 3: trend_long, 4: trend_short]
        y_stacked_fold = []
        for i, idx in enumerate(test_idx):
            lbl = raw_labels[idx]
            reg = regime_labels_list[i]
            y_stacked_fold.append(map_meta_class(feature_df.iloc[idx], lbl, reg))
        y_stacked_fold = np.array(y_stacked_fold)
        
        # Form stacked feature matrix for this fold: (len(test_idx), 8)
        X_stacked_fold = np.column_stack([
            regime_probs_oos,  # 3 cols
            entry_probs_oos,   # 3 cols
            strength_p50_oos,  # 1 col
            reversal_prob_oos  # 1 col
        ])
        
        # Train Meta Stacker online on prior accumulated out-of-fold data to generate unbiased test outputs
        meta_action_oos = []
        if len(stacked_X_list) > 50:
            train_stacker_X = np.vstack(stacked_X_list)
            train_stacker_y = np.concatenate(stacked_y_list)
            
            temp_meta = MetaDecisionEngine()
            temp_meta.fit(train_stacker_X, train_stacker_y)
            
            # Predict OOS
            for i in range(len(X_test)):
                dec = temp_meta.decide(
                    regime_probs=regime_probs_oos[i],
                    entry_probs=entry_probs_oos[i],
                    strength_p50=strength_p50_oos[i],
                    reversal_prob=reversal_prob_oos[i],
                    current_time=datetime.datetime.now(datetime.timezone.utc),
                    news_events=[],
                    active_positions_count=0
                )
                # Map action back to meta integer class for performance testing
                action_to_class = {"no_trade": 0, "scalp_long": 1, "scalp_short": 2, "trend_long": 3, "trend_short": 4}
                meta_action_oos.append(action_to_class.get(dec["action"], 0))
        else:
            # Fallback to no-trade if stacker training data is too small
            meta_action_oos = [0] * len(X_test)
            
        # Accumulate predictions
        stacked_X_list.append(X_stacked_fold)
        stacked_y_list.append(y_stacked_fold)
        
        all_oos_y_true.extend(y_test)
        all_oos_entry_preds.extend(entry_preds_oos)
        all_oos_meta_preds.extend(meta_action_oos)
        all_oos_entry_probs.extend(entry_probs_oos)
        
        logger.info(f"Fold {fold_idx} base model evaluations complete.")
        fold_idx += 1
        
    # Convert lists to arrays
    all_oos_y_true = np.array(all_oos_y_true)
    all_oos_entry_preds = np.array(all_oos_entry_preds)
    all_oos_meta_preds = np.array(all_oos_meta_preds)
    all_oos_entry_probs = np.array(all_oos_entry_probs)
    
    # 6. Fit final models on complete dataset
    logger.info("Fitting final production models on full historical dataset...")
    
    # Final HMM
    final_regime = RegimeDetector()
    final_regime.fit(feature_df)
    
    # Final Entry Model
    final_split = int(len(X) * 0.85)
    final_entry = EntryModel()
    final_entry.fit_and_calibrate(
        X.iloc[:final_split], y[:final_split], weights.iloc[:final_split].values,
        X.iloc[final_split:], y[final_split:]
    )
    
    # Final Strength Model
    final_strength = StrengthModel()
    y_strength_full = final_strength.compute_targets(feature_df, labels_df)
    final_strength.fit(X, y_strength_full, sample_weight=weights.values)
    
    # Final Reversal Model
    final_reversal = ReversalModel()
    y_reversal_full = final_reversal.compute_targets(reversal_feat_df)
    final_reversal.fit(reversal_feat_df[reversal_features_cols], y_reversal_full)
    
    # Final Meta Decision Engine Stacker
    # Train stacker on ALL compiled out-of-fold base predictions
    train_stacker_X_full = np.vstack(stacked_X_list)
    train_stacker_y_full = np.concatenate(stacked_y_list)
    
    final_meta = MetaDecisionEngine()
    final_meta.fit(train_stacker_X_full, train_stacker_y_full)
    
    # 7. Register all models in ModelRegistry
    registry = ModelRegistry(registry_dir="models")
    dummy_metrics = {"n_samples": len(df)}
    
    registry.save_model("regime_hmm", final_regime, df, dummy_metrics, version=1)
    registry.save_model("entry_lgb", final_entry, df, dummy_metrics, version=1)
    registry.save_model("strength_lgb", final_strength, df, dummy_metrics, version=1)
    registry.save_model("reversal_lgb", final_reversal, df, dummy_metrics, version=1)
    registry.save_model("meta_stacker", final_meta, df, dummy_metrics, version=1)
    
    # 8. Report risk-adjusted performance comparison
    # Map meta predictions back to entry signals:
    # classes: 1 (scalp_long), 3 (trend_long) -> 1 (BUY)
    # classes: 2 (scalp_short), 4 (trend_short) -> 2 (SELL)
    # class: 0 -> 0 (HOLD)
    meta_signals = np.zeros(len(all_oos_meta_preds))
    meta_signals[(all_oos_meta_preds == 1) | (all_oos_meta_preds == 3)] = 1
    meta_signals[(all_oos_meta_preds == 2) | (all_oos_meta_preds == 4)] = 2
    
    # Entry baseline signals
    entry_signals = all_oos_entry_preds
    
    # Compute test indices for OOS evaluation (excluding the initial train fold size)
    segment_size = len(feature_df) // (n_folds + 1)
    test_indices = list(range(segment_size, len(all_oos_y_true) + segment_size))
    
    # Compute performance metrics
    entry_perf = compute_fold_performance(feature_df, entry_signals, test_indices)
    meta_perf = compute_fold_performance(feature_df, meta_signals, test_indices)
    
    logger.info("==============================================================")
    logger.info("📈 BACKTEST PERFORMANCE COMPARISON (Phase 2 vs Phase 3 Stacker)")
    logger.info("==============================================================")
    logger.info(f"Entry-Only Baseline (Phase 2): Sharpe = {entry_perf['sharpe']:.4f} | Max Drawdown = {entry_perf['max_drawdown']:.6f}")
    logger.info(f"Stacked Meta Engine (Phase 3): Sharpe = {meta_perf['sharpe']:.4f} | Max Drawdown = {meta_perf['max_drawdown']:.6f}")
    logger.info("==============================================================")
    
    # Plot Calibration Curve for Entry Model Long Direction
    y_true_long = (all_oos_y_true == 1).astype(int)
    y_prob_long = all_oos_entry_probs[:, 1]
    prob_true, prob_pred = calibration_curve(y_true_long, y_prob_long, n_bins=5)
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibrated LightGBM")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Predicted Probability (Long)")
    plt.ylabel("Actual Frequency (Long)")
    plt.title("Isotonic Probability Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("calibration_curve.png")
    plt.close()
    
    logger.info("Pipeline execution complete. Registered final model stack.")

if __name__ == "__main__":
    main()
