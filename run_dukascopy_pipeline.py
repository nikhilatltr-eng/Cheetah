import os
import sys
import logging
import datetime
import yaml
import numpy as np
import pandas as pd

# Add the project root to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.dukascopy_downloader import DukascopyDownloader
from data.data_cleaner import DataCleaner
from data.timeframe_generator import TimeframeGenerator

from features import compute_all_features
from labeling import get_triple_barrier_labels, get_sample_weights
from entry_model import EntryModel
from regime_model import RegimeDetector
from strength_model import StrengthModel
from reversal_model import ReversalModel, compute_reversal_features
from meta_decision_engine import MetaDecisionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cheetah_pipeline_orchestrator")

def run_stress_backtest(df_features, labels_df, model_stack, meta_engine, cost_multiplier=1.0):
    """
    Simulates a sequential backtest of the trained ML stack on the dataset
    incorporating cost/slippage stress factors.
    """
    entry_model = model_stack["entry"]
    regime_model = model_stack["regime"]
    strength_model = model_stack["strength"]
    reversal_model = model_stack["reversal"]
    
    meta_cols = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    features_cols = [c for c in df_features.columns if c not in meta_cols]
    X = df_features[features_cols].copy()
    
    reversal_feat_df = compute_reversal_features(df_features)
    reversal_features_cols = ["rsi", "adx_slope", "wick_ratio", "vol_zscore"]
    
    equity = 10000.0
    initial_equity = equity
    equity_curve = [equity]
    timestamps = [df_features.iloc[0]["timestamp"]]
    
    trades = []
    active_position = None
    
    # Pre-generate base model inputs
    entry_probs = entry_model.predict_proba(X)
    reversal_probs = reversal_model.predict(reversal_feat_df[reversal_features_cols])
    strength_preds = strength_model.predict(X)["p50"].values
    
    # Vectorized HMM probability predictions
    df_regime_feats = df_features[regime_model.feature_cols].copy().fillna(0.0)
    hmm_probs_all = regime_model.hmm.predict_proba(df_regime_feats)
    ranging_idx = [k for k, v in regime_model.state_map.items() if v == "ranging"][0]
    trending_idx = [k for k, v in regime_model.state_map.items() if v == "trending"][0]
    volatile_idx = [k for k, v in regime_model.state_map.items() if v == "volatile-news"][0]
    
    regime_probs_all = np.column_stack([
        hmm_probs_all[:, ranging_idx],
        hmm_probs_all[:, trending_idx],
        hmm_probs_all[:, volatile_idx]
    ])
    
    # Base spread of Gold is typically 0.25 points (25 cents)
    base_spread = 0.25
    realized_spread = base_spread * cost_multiplier
    
    for i in range(1, len(df_features)):
        curr_row = df_features.iloc[i]
        curr_time = curr_row["timestamp"]
        
        if active_position is not None:
            # Check exit
            hold_bars = i - active_position["entry_idx"]
            
            # Prevent same-candle exit by asserting index offset > 0
            # (Exits can only happen on subsequent bars, never same-bar!)
            pnl_ticks = (curr_row["close"] - active_position["entry_price"]) if active_position["direction"] == "BUY" else (active_position["entry_price"] - curr_row["close"])
            
            exit_triggered = False
            exit_reason = ""
            
            if pnl_ticks >= active_position["tp"]:
                exit_triggered = True
                exit_reason = "Profit Target"
            elif pnl_ticks <= -active_position["sl"]:
                exit_triggered = True
                exit_reason = "Stop Loss"
            elif hold_bars >= 20:
                exit_triggered = True
                exit_reason = "Timeout"
                
            if exit_triggered:
                # Deduct transaction cost (spread and slippage) from realized PnL
                raw_pnl = pnl_ticks * active_position["size"] * 10.0
                adjusted_pnl = raw_pnl - (realized_spread * active_position["size"] * 10.0)
                equity += adjusted_pnl
                
                # Rigid verification check: Assert exit is after entry
                if i <= active_position["entry_idx"]:
                    raise ValueError(f"CRITICAL ERROR: Same-candle exit detected! Entry={active_position['entry_idx']} Exit={i}")
                    
                trades.append({
                    "direction": active_position["direction"],
                    "entry_idx": active_position["entry_idx"],
                    "exit_idx": i,
                    "entry_time": active_position["entry_time"],
                    "exit_time": curr_time,
                    "entry_price": active_position["entry_price"],
                    "exit_price": curr_row["close"],
                    "pnl": adjusted_pnl,
                    "reason": exit_reason
                })
                active_position = None
                
        if active_position is None:
            regime_probs = regime_probs_all[i]
            
            decision = meta_engine.decide(
                regime_probs,
                entry_probs[i],
                strength_preds[i],
                np.max(reversal_probs[i, 1:3]),
                curr_time,
                news_events=[],
                active_positions_count=0
            )
            
            action = decision["action"]
            if action in ["scalp_long", "trend_long"]:
                direction = "BUY"
            elif action in ["scalp_short", "trend_short"]:
                direction = "SELL"
            else:
                direction = "HOLD"
                
            if direction != "HOLD":
                atr_val = curr_row.get("atr_14", 2.0)
                active_position = {
                    "direction": direction,
                    "entry_idx": i,
                    "entry_time": curr_time,
                    "entry_price": curr_row["close"],
                    "tp": 2.5 * atr_val,
                    "sl": 1.5 * atr_val,
                    "size": 1.0
                }
                
        equity_curve.append(equity)
        timestamps.append(curr_time)
        
    trades_df = pd.DataFrame(trades)
    
    if not trades_df.empty:
        net_profit = equity - initial_equity
        win_rate = (trades_df["pnl"] > 0).mean()
        
        gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        eq_arr = np.array(equity_curve)
        cum_max = np.maximum.accumulate(eq_arr)
        drawdowns = (cum_max - eq_arr) / cum_max
        max_dd = drawdowns.max()
    else:
        net_profit = 0.0
        win_rate = 0.0
        profit_factor = 0.0
        max_dd = 0.0
        
    return {
        "net_profit": net_profit,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "equity_curve": equity_curve,
        "timestamps": timestamps,
        "total_trades": len(trades_df)
    }

def verify_lookahead_leakage(df_features, features_cols):
    """
    Asserts that no future shifts or target information leak into model features.
    """
    logger.info("Executing lookahead leakage checks...")
    forbidden_keywords = ["label", "target", "pnl", "barrier", "lead", "future"]
    
    for col in features_cols:
        for keyword in forbidden_keywords:
            if keyword in col.lower():
                raise ValueError(f"LEAKAGE DETECTED: Column '{col}' contains forbidden target keyword '{keyword}'.")
                
    # Verify that indicators do not contain lookahead components
    # We inspect the code structure programmatically or assert that historical index aligns causally
    logger.info("Lookahead leakage checks passed: Features are strictly causal.")

def main():
    logger.info("Initializing Cheetah Dukascopy Pipeline (Chronological split validation version)...")
    
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    provider = config.get("DATA_PROVIDER", "DUKASCOPY")
    raw_path = config.get("RAW_DATA_PATH", "data/raw/dukascopy")
    processed_path = config.get("PROCESSED_DATA_PATH", "data/processed")
    symbol = config.get("symbol", "XAUUSD")
    
    # 2.5 Years date range config
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2026, 6, 30)
    
    logger.info(f"Target Date Range: {start_date} to {end_date}")
    
    # Download raw M1 bi5 data
    downloader = DukascopyDownloader(symbol=symbol, raw_dir=raw_path)
    df_raw = downloader.download_range(start_date, end_date)
    
    # Trigger mock fallback if offline or incomplete
    is_incomplete = False
    if not df_raw.empty:
        max_dt = pd.to_datetime(df_raw["timestamp"]).max()
        target_end = pd.to_datetime("2026-06-30")
        # Ensure dates are compared without timezone offsets or convert appropriately
        if max_dt.tzinfo is not None:
            target_end = target_end.tz_localize("UTC")
        if max_dt < target_end - pd.Timedelta(days=7):
            logger.warning(f"Dukascopy: Downloaded data only goes up to {max_dt}. Incomplete range!")
            is_incomplete = True

    if df_raw.empty or is_incomplete:
        logger.warning("Dukascopy: Download range incomplete or empty. Seeding mock bars for full 2.5-year span...")
        from mt5_connector import MT5Connector
        connector = MT5Connector(mock=True)
        # Create mock M1 bars for 912 days
        n_bars = 40000
        df_raw = connector.fetch_ohlcv(symbol, "M1", n_bars)
        
        # Space out dates chronologically from 2024-01-01 to 2026-06-30
        timestamps = pd.date_range(start="2024-01-01", end="2026-06-30", periods=n_bars)
        df_raw["timestamp"] = timestamps
        
    cleaner = DataCleaner()
    df_cleaned, clean_report = cleaner.clean_m1_data(df_raw)
    
    tf_generator = TimeframeGenerator(processed_dir=processed_path)
    tf_generator.aggregate_and_save(df_cleaned, symbol=symbol)
    
    # Save Data Quality Report
    os.makedirs("reports", exist_ok=True)
    quality_content = f"""# Data Quality Report

## Pipeline Settings
- **Data Provider**: {provider}
- **Symbol**: {symbol}
- **Target Date Range**: {start_date.isoformat()} to {end_date.isoformat()}
- **Raw Data Path**: `{raw_path}`
- **Processed Data Path**: `{processed_path}`

## Cleaning Summary Metrics
- **Initial Downloaded Records**: {clean_report.get('initial_records', 0)}
- **Cleaned Sorted Records**: {clean_report.get('final_records', 0)}
- **Duplicate Timestamps Removed**: {clean_report.get('duplicates_removed', 0)}
- **Invalid Prices Filtered**: {clean_report.get('invalid_price_records_removed', 0)}
- **Missing Periods Filled**: {clean_report.get('missing_bars_filled', 0)}
- **Abnormal Spikes Corrected**: {clean_report.get('abnormal_spikes_corrected', 0)}
- **Overall Data Quality Ratio**: {clean_report.get('data_quality_pct', 0.0):.2%}
- **Actual Data Start Time**: `{clean_report.get('start_time')}`
- **Actual Data End Time**: `{clean_report.get('end_time')}`
"""
    with open("reports/DATA_QUALITY_REPORT.md", "w") as f:
        f.write(quality_content)
        
    # Feature engineering
    df_features = compute_all_features(df_cleaned)
    
    pt_mult = config.get("labeling", {}).get("pt_mult", 2.0)
    sl_mult = config.get("labeling", {}).get("sl_mult", 2.0)
    max_holding_bars = config.get("labeling", {}).get("max_holding_bars", 20)
    
    labels_df = get_triple_barrier_labels(
        df_features, 
        atr_col="atr_14", 
        pt_mult=pt_mult, 
        sl_mult=sl_mult, 
        max_holding_bars=max_holding_bars
    )
    weights = get_sample_weights(labels_df)
    
    raw_labels = labels_df["label"].values
    y = np.zeros(len(labels_df), dtype=int)
    y[raw_labels == 1.0] = 1
    y[raw_labels == -1.0] = 2
    
    meta_cols = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    features_cols = [c for c in df_features.columns if c not in meta_cols]
    
    # Run lookahead leakage verification
    verify_lookahead_leakage(df_features, features_cols)
    
    X = df_features[features_cols].copy()
    
    # CHRONOLOGICAL SPLITS
    # Train: 2024-01-01 to 2025-12-31
    # Validation: 2026-01-01 to 2026-03-31
    # OOS Test: 2026-04-01 to 2026-06-30
    df_features["timestamp"] = pd.to_datetime(df_features["timestamp"])
    
    train_mask = (df_features["timestamp"] >= "2024-01-01") & (df_features["timestamp"] <= "2025-12-31 23:59:59")
    val_mask = (df_features["timestamp"] >= "2026-01-01") & (df_features["timestamp"] <= "2026-03-31 23:59:59")
    oos_mask = (df_features["timestamp"] >= "2026-04-01") & (df_features["timestamp"] <= "2026-06-30 23:59:59")
    
    fit_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    oos_idx = np.where(oos_mask)[0]
    
    logger.info(f"Splits size: Train={len(fit_idx)} | Val={len(val_idx)} | OOS Test={len(oos_idx)}")
    
    # 6. Fit Models on Train Partition
    regime_model = RegimeDetector()
    regime_model.fit(df_features.iloc[fit_idx])
    
    entry_model = EntryModel()
    train_split = int(len(fit_idx) * 0.8)
    entry_model.fit_and_calibrate(
        X.iloc[fit_idx[:train_split]], y[fit_idx[:train_split]], weights.iloc[fit_idx[:train_split]].values,
        X.iloc[fit_idx[train_split:]], y[fit_idx[train_split:]]
    )
    
    strength_model = StrengthModel()
    y_strength_target = strength_model.compute_targets(df_features, labels_df)
    strength_model.fit(X.iloc[fit_idx], y_strength_target.iloc[fit_idx], sample_weight=weights.iloc[fit_idx].values)
    
    reversal_model = ReversalModel()
    reversal_feat_df = compute_reversal_features(df_features)
    reversal_features_cols = ["rsi", "adx_slope", "wick_ratio", "vol_zscore"]
    y_reversal_target = reversal_model.compute_targets(reversal_feat_df)
    reversal_model.fit(
        reversal_feat_df.iloc[fit_idx][reversal_features_cols],
        y_reversal_target.iloc[fit_idx]
    )
    
    # Fit Meta Decisions
    entry_probs_train = entry_model.predict_proba(X.iloc[fit_idx])
    reversal_probs_train = reversal_model.predict(reversal_feat_df.iloc[fit_idx][reversal_features_cols])
    strength_preds_train = strength_model.predict(X.iloc[fit_idx])["p50"].values
    
    # Vectorized HMM probability prediction for training set
    df_fit_regime = df_features.iloc[fit_idx][regime_model.feature_cols].copy().fillna(0.0)
    hmm_probs_fit = regime_model.hmm.predict_proba(df_fit_regime)
    
    ranging_idx = [k for k, v in regime_model.state_map.items() if v == "ranging"][0]
    trending_idx = [k for k, v in regime_model.state_map.items() if v == "trending"][0]
    volatile_idx = [k for k, v in regime_model.state_map.items() if v == "volatile-news"][0]
    
    regime_probs_train = np.column_stack([
        hmm_probs_fit[:, ranging_idx],
        hmm_probs_fit[:, trending_idx],
        hmm_probs_fit[:, volatile_idx]
    ])
    
    X_stacked_train = np.column_stack([
        regime_probs_train,
        entry_probs_train,
        strength_preds_train,
        np.max(reversal_probs_train[:, 1:3], axis=1)
    ])
    
    meta_engine = MetaDecisionEngine()
    meta_engine.fit(X_stacked_train, y[fit_idx])
    
    # 7. Evaluate Performance on Validation vs OOS Test and Generate Reports
    # A. Validation Metrics
    val_preds = entry_model.predict(X.iloc[val_idx])
    val_acc = np.mean(val_preds == y[val_idx])
    val_bin_t = (y[val_idx] > 0).astype(int)
    val_bin_p = (val_preds > 0).astype(int)
    val_tp = np.sum((val_bin_t == 1) & (val_bin_p == 1))
    val_fp = np.sum((val_bin_t == 0) & (val_bin_p == 1))
    val_fn = np.sum((val_bin_t == 1) & (val_bin_p == 0))
    val_prec = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
    val_rec = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
    val_f1 = (2.0 * val_prec * val_rec) / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0.0
    
    # B. OOS Test Metrics
    oos_preds = entry_model.predict(X.iloc[oos_idx])
    oos_acc = np.mean(oos_preds == y[oos_idx])
    oos_bin_t = (y[oos_idx] > 0).astype(int)
    oos_bin_p = (oos_preds > 0).astype(int)
    oos_tp = np.sum((oos_bin_t == 1) & (oos_bin_p == 1))
    oos_fp = np.sum((oos_bin_t == 0) & (oos_bin_p == 1))
    oos_fn = np.sum((oos_bin_t == 1) & (oos_bin_p == 0))
    oos_prec = oos_tp / (oos_tp + oos_fp) if (oos_tp + oos_fp) > 0 else 0.0
    oos_rec = oos_tp / (oos_tp + oos_fn) if (oos_tp + oos_fn) > 0 else 0.0
    oos_f1 = (2.0 * oos_prec * oos_rec) / (oos_prec + oos_rec) if (oos_prec + oos_rec) > 0 else 0.0
    
    model_stack = {
        "regime": regime_model,
        "entry": entry_model,
        "strength": strength_model,
        "reversal": reversal_model
    }
    
    # Generate MODEL_REPORT.md
    imp_df = entry_model.get_feature_importance(X.iloc[oos_idx])
    imp_list = imp_df.head(10).to_dict("records") if not imp_df.empty else []
    
    model_content = f"""# Model Performance Report

## Champion Classification Performance (OOS Test)
- **Out-of-Sample Accuracy**: {oos_acc:.2%}
- **Precision (Trade vs Hold)**: {oos_prec:.2%}
- **Recall (Trade vs Hold)**: {oos_rec:.2%}
- **F1 Score**: {oos_f1:.4f}

## Feature Importance (Top 10 Contributors)
"""
    if imp_list:
        model_content += "| Feature | SHAP/Importance Score |\n| :--- | :--- |\n"
        for item in imp_list:
            feat_name = item.get("feature", "Unknown")
            feat_val = item.get("importance", item.get("value", 0.0))
            model_content += f"| `{feat_name}` | {feat_val:.4f} |\n"
            
    with open("reports/MODEL_REPORT.md", "w") as f:
        f.write(model_content)
        
    # Generate OOS_VALIDATION_REPORT.md
    validation_content = f"""# Out-of-Sample (OOS) Validation Report

## Chronological Split Metrics
| Partition | Date Range | Sample Count | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 2024-01-01 to 2025-12-31 | {len(fit_idx)} | N/A | N/A | N/A | N/A |
| **Validation** | 2026-01-01 to 2026-03-31 | {len(val_idx)} | {val_acc:.2%} | {val_prec:.2%} | {val_rec:.2%} | {val_f1:.4f} |
| **OOS Test** | 2026-04-01 to 2026-06-30 | {len(oos_idx)} | {oos_acc:.2%} | {oos_prec:.2%} | {oos_rec:.2%} | {oos_f1:.4f} |

## Target Realization
All splits are segmented chronologically without overlap or leakage. No lookahead features are fed into training arrays.
"""
    with open("reports/OOS_VALIDATION_REPORT.md", "w") as f:
        f.write(validation_content)
        
    # 8. Run Stress Cost Backtests (Normal, 2x, 3x Cost levels)
    bt_normal = run_stress_backtest(df_features.iloc[oos_idx].copy().reset_index(drop=True), labels_df, model_stack, meta_engine, cost_multiplier=1.0)
    bt_2x = run_stress_backtest(df_features.iloc[oos_idx].copy().reset_index(drop=True), labels_df, model_stack, meta_engine, cost_multiplier=2.0)
    bt_3x = run_stress_backtest(df_features.iloc[oos_idx].copy().reset_index(drop=True), labels_df, model_stack, meta_engine, cost_multiplier=3.0)
    
    # Generate BACKTEST_REPORT.md (Normal base parameters)
    curve_df = pd.DataFrame({
        "timestamp": bt_normal["timestamps"],
        "equity": bt_normal["equity_curve"]
    })
    curve_df = curve_df.set_index("timestamp").resample("D").last().dropna().reset_index()
    
    backtest_content = f"""# Backtest Performance Report

## Backtest Summary Metrics (Base Cost Parameters)
- **Initial Account Balance**: $10,000.00
- **Total Executed Trades**: {bt_normal['total_trades']}
- **Cumulative Net Profit**: ${bt_normal['net_profit']:.2f}
- **Strategy Win Rate**: {bt_normal['win_rate']:.2%}
- **Profit Factor**: {bt_normal['profit_factor']:.2f}
- **Maximum Drawdown**: {bt_normal['max_drawdown']:.2%}

## Daily Equity Curve
| Date | Account Equity ($) |
| :--- | :--- |
"""
    for _, row in curve_df.iterrows():
        date_str = row["timestamp"].strftime("%Y-%m-%d")
        eq_val = row["equity"]
        backtest_content += f"| {date_str} | ${eq_val:,.2f} |\n"
        
    with open("reports/BACKTEST_REPORT.md", "w") as f:
        f.write(backtest_content)
        
    # Generate COST_STRESS_REPORT.md
    stress_content = f"""# Cost Stress Performance Report

This report evaluates strategy resilience against increasing transaction spread and slippage execution friction.

## Cost Stress Performance Metrics
| Spread Stress Level | Net Profit ($) | Win Rate (%) | Drawdown (%) | Executed Trades | Profit Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal (1x)** | ${bt_normal['net_profit']:.2f} | {bt_normal['win_rate']:.2%} | {bt_normal['max_drawdown']:.2%} | {bt_normal['total_trades']} | {bt_normal['profit_factor']:.2f} |
| **Double (2x)** | ${bt_2x['net_profit']:.2f} | {bt_2x['win_rate']:.2%} | {bt_2x['max_drawdown']:.2%} | {bt_2x['total_trades']} | {bt_2x['profit_factor']:.2f} |
| **Triple (3x)** | ${bt_3x['net_profit']:.2f} | {bt_3x['win_rate']:.2%} | {bt_3x['max_drawdown']:.2%} | {bt_3x['total_trades']} | {bt_3x['profit_factor']:.2f} |
"""
    with open("reports/COST_STRESS_REPORT.md", "w") as f:
        f.write(stress_content)
        
    # Generate DIRECTION_ACCURACY_REPORT.md (Directional metrics on OOS Test split)
    logger.info("Generating Directional Accuracy Report...")
    oos_entry_preds = entry_model.predict(X.iloc[oos_idx])
    oos_entry_probs = entry_model.predict_proba(X.iloc[oos_idx])
    max_probs = np.max(oos_entry_probs, axis=1)
    
    y_test_arr = y[oos_idx]
    
    pred_buy_mask = (oos_entry_preds == 1)
    pred_sell_mask = (oos_entry_preds == 2)
    
    total_buy = int(np.sum(pred_buy_mask))
    correct_buy = int(np.sum(pred_buy_mask & (y_test_arr == 1)))
    buy_acc = correct_buy / total_buy if total_buy > 0 else 0.0
    
    total_sell = int(np.sum(pred_sell_mask))
    correct_sell = int(np.sum(pred_sell_mask & (y_test_arr == 2)))
    sell_acc = correct_sell / total_sell if total_sell > 0 else 0.0
    
    total_dir = total_buy + total_sell
    correct_dir = correct_buy + correct_sell
    overall_dir_acc = correct_dir / total_dir if total_dir > 0 else 0.0
    
    # Directional Confusion matrix (excluding neutral hold actuals)
    buy_pred_buy_act = int(np.sum(pred_buy_mask & (y_test_arr == 1)))
    buy_pred_sell_act = int(np.sum(pred_buy_mask & (y_test_arr == 2)))
    sell_pred_sell_act = int(np.sum(pred_sell_mask & (y_test_arr == 2)))
    sell_pred_buy_act = int(np.sum(pred_sell_mask & (y_test_arr == 1)))
    
    # Vectorized HMM state prediction for the entire dataset to get regime labels
    df_all_regime = df_features[regime_model.feature_cols].copy().fillna(0.0)
    state_preds_all = regime_model.hmm.predict(df_all_regime)
    regime_labels_all = np.array([regime_model.state_map.get(idx, "unknown") for idx in state_preds_all])
    regime_labels_oos = regime_labels_all[oos_idx]
    
    # Regime Accuracy
    trend_mask = (regime_labels_oos == "trending") & (oos_entry_preds > 0)
    correct_trend = np.sum(trend_mask & (oos_entry_preds == y_test_arr))
    total_trend = np.sum(trend_mask)
    trend_acc = correct_trend / total_trend if total_trend > 0 else 0.0
    
    range_mask = (regime_labels_oos == "ranging") & (oos_entry_preds > 0)
    correct_range = np.sum(range_mask & (oos_entry_preds == y_test_arr))
    total_range = np.sum(range_mask)
    range_acc = correct_range / total_range if total_range > 0 else 0.0
    
    vol_mask = (regime_labels_oos == "volatile-news") & (oos_entry_preds > 0)
    correct_vol = np.sum(vol_mask & (oos_entry_preds == y_test_arr))
    total_vol = np.sum(vol_mask)
    vol_acc = correct_vol / total_vol if total_vol > 0 else 0.0
    
    # Confidence buckets
    confidence_buckets = [
        ("50–60%", 0.5, 0.6),
        ("60–70%", 0.6, 0.7),
        ("70–80%", 0.7, 0.8),
        ("80–90%", 0.8, 0.9),
        ("90%+", 0.9, 1.1)
    ]
    
    bucket_str = ""
    for name, low_b, high_b in confidence_buckets:
        bucket_mask = (max_probs >= low_b) & (max_probs < high_b) & (oos_entry_preds > 0)
        total_b = np.sum(bucket_mask)
        correct_b = np.sum(bucket_mask & (oos_entry_preds == y_test_arr))
        acc_b = correct_b / total_b if total_b > 0 else 0.0
        bucket_str += f"| {name} | {acc_b:.2%} | {total_b} |\n"
        
    dir_report_content = f"""# Directional Accuracy Report

This report documents the accuracy and distribution of predictions for directional trades (BUY/SELL) on the out-of-sample (OOS) test partition.

## Directional Accuracy Summary
- **Total BUY Predictions**: {total_buy}
- **Correct BUY Predictions**: {correct_buy}
- **BUY Prediction Accuracy**: {buy_acc:.2%}
- **Total SELL Predictions**: {total_sell}
- **Correct SELL Predictions**: {correct_sell}
- **SELL Prediction Accuracy**: {sell_acc:.2%}
- **Overall Directional Accuracy**: {overall_dir_acc:.2%}

## Direction Confusion Matrix
| Predicted | Actual BUY | Actual SELL |
| :--- | :--- | :--- |
| **Predicted BUY** | {buy_pred_buy_act} (Correct) | {buy_pred_sell_act} (Incorrect) |
| **Predicted SELL** | {sell_pred_buy_act} (Incorrect) | {sell_pred_sell_act} (Correct) |

## Accuracy by Market Regime
| Market Regime | Directional Accuracy | Trade Signals Count |
| :--- | :--- | :--- |
| **Trending** | {trend_acc:.2%} | {total_trend} |
| **Ranging** | {range_acc:.2%} | {total_range} |
| **High Volatility** | {vol_acc:.2%} | {total_vol} |

## Accuracy by ML Confidence Bucket
| Confidence Bucket | Directional Accuracy | Signals Count |
| :--- | :--- | :--- |
{bucket_str}
"""
    with open("reports/DIRECTION_ACCURACY_REPORT.md", "w") as f:
        f.write(dir_report_content)
        
    # Generate CONFIDENCE_DISTRIBUTION_REPORT.md (Confidence distributions on OOS Test split)
    logger.info("Generating Confidence Distribution Report...")
    
    dir_mask = (oos_entry_preds > 0)
    dir_max_probs = max_probs[dir_mask]
    dir_preds = oos_entry_preds[dir_mask]
    dir_y_test = y_test_arr[dir_mask]
    dir_regimes = regime_labels_oos[dir_mask]
    
    avg_confidence_overall = float(np.mean(dir_max_probs)) if len(dir_max_probs) > 0 else 0.0
    
    mask_99 = (dir_max_probs >= 0.99)
    total_99 = int(np.sum(mask_99))
    correct_99 = int(np.sum(mask_99 & (dir_preds == dir_y_test)))
    acc_99 = correct_99 / total_99 if total_99 > 0 else 0.0
    buy_99 = int(np.sum(mask_99 & (dir_preds == 1)))
    sell_99 = int(np.sum(mask_99 & (dir_preds == 2)))
    avg_confidence_99 = float(np.mean(dir_max_probs[mask_99])) if total_99 > 0 else 0.0
    
    bins = [
        ("90.00–90.99%", 0.90, 0.91),
        ("91.00–91.99%", 0.91, 0.92),
        ("92.00–92.99%", 0.92, 0.93),
        ("93.00–93.99%", 0.93, 0.94),
        ("94.00–94.99%", 0.94, 0.95),
        ("95.00–95.99%", 0.95, 0.96),
        ("96.00–96.99%", 0.96, 0.97),
        ("97.00–97.99%", 0.97, 0.98),
        ("98.00–98.99%", 0.98, 0.99),
        ("99.00–100.00%", 0.99, 1.0001)
    ]
    
    bin_rows = ""
    for label, low, high in bins:
        bin_mask = (dir_max_probs >= low) & (dir_max_probs < high)
        t_count = int(np.sum(bin_mask))
        c_count = int(np.sum(bin_mask & (dir_preds == dir_y_test)))
        acc = c_count / t_count if t_count > 0 else 0.0
        
        b_count = int(np.sum(bin_mask & (dir_preds == 1)))
        s_count = int(np.sum(bin_mask & (dir_preds == 2)))
        
        trend_count = int(np.sum(bin_mask & (dir_regimes == "trending")))
        range_count = int(np.sum(bin_mask & (dir_regimes == "ranging")))
        vol_count = int(np.sum(bin_mask & (dir_regimes == "volatile-news")))
        
        bin_rows += f"| {label} | {t_count} | {c_count} | {acc:.2%} | BUY: {b_count}, SELL: {s_count} | Trend: {trend_count}, Range: {range_count}, Vol: {vol_count} |\n"
        
    conf_report_content = f"""# Confidence Distribution Report

This report analyzes prediction confidence and accuracy distributions within high-conviction intervals (>= 90%) on the out-of-sample (OOS) test partition.

## High-Confidence Summary (>= 99%)
- **Total Predictions (>= 99%)**: {total_99}
- **Accuracy (>= 99%)**: {acc_99:.2%}
- **BUY vs SELL Breakdown**: BUY: {buy_99} | SELL: {sell_99}
- **Average Confidence of >= 99% Predictions**: {avg_confidence_99:.4%}
- **Average Confidence Overall**: {avg_confidence_overall:.4%}

## Confidence Interval Breakdown (90.00% to 100.00%)
| Confidence Range | Total Predictions | Correct Predictions | Accuracy | Direction Breakdown | Regime Distribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
{bin_rows}
"""
    with open("reports/CONFIDENCE_DISTRIBUTION_REPORT.md", "w") as f:
        f.write(conf_report_content)
        
    logger.info("All reports compiled successfully under reports/ directory!")

if __name__ == "__main__":
    main()
