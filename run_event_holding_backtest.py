import logging
import os
import numpy as np
import pandas as pd
import datetime
from data.dukascopy_downloader import DukascopyDownloader
from data.data_cleaner import DataCleaner
from data.timeframe_generator import TimeframeGenerator
from features import compute_all_features
from labeling import get_triple_barrier_labels, get_sample_weights
from entry_model import EntryModel
from regime_model import RegimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("event_holding_backtest")

# Realistic execution friction parameters for Gold (XAUUSD)
DEFAULT_SPREAD = 0.15      # 15 points ($0.15) default spread
DEFAULT_SLIPPAGE = 0.05    # 5 points ($0.05) execution slippage penalty

def calculate_drawdown(pnls):
    if not pnls:
        return 0.0
    equity = 10000.0 + np.cumsum(pnls)
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd

def simulate_policy(policy_name, threshold, oos_idx, max_probs, oos_entry_preds, close_oos, high_oos, low_oos, ema21_oos, ema55_oos, atr_oos, regime_labels_oos, time_oos, volume_oos):
    """
    Simulates position holding for a specific policy.
    One open position only. No re-entry while active. Only enter on active market bars.
    """
    trades = []
    active_position = None
    n_oos = len(oos_idx)
    
    for i in range(n_oos):
        conf = max_probs[i]
        pred = oos_entry_preds[i]
        vol = volume_oos[i]
        # Signal is present only during active market hours
        is_signal = (pred == 1) and (conf >= threshold) and (vol > 0.0)
        
        close_p = close_oos[i]
        high_p = high_oos[i]
        low_p = low_oos[i]
        ema21 = ema21_oos[i]
        ema55 = ema55_oos[i]
        atr = atr_oos[i]
        regime = regime_labels_oos[i]
        
        if active_position is None:
            if is_signal:
                # Enter Long at Ask
                active_position = {
                    "start_idx": i,
                    "start_time": time_oos[i],
                    "entry_price": close_p + DEFAULT_SPREAD + DEFAULT_SLIPPAGE,
                    "entry_atr": atr,
                    "highest_bid_seen": close_p,
                    "signals_during_trade": 1
                }
        else:
            # Increment raw signals count
            if (pred == 1) and (conf >= threshold):
                active_position["signals_during_trade"] += 1
                
            # Update trailing metric
            active_position["highest_bid_seen"] = max(active_position["highest_bid_seen"], close_p)
            
            # Check exit conditions based on policy
            exit_triggered = False
            exit_price = close_p - DEFAULT_SLIPPAGE  # Default exit price
            
            if policy_name == "EMA21 exit":
                if close_p < ema21:
                    exit_triggered = True
                    
            elif policy_name == "EMA55 exit":
                if close_p < ema55:
                    exit_triggered = True
                    
            elif policy_name == "ATR trailing stop":
                trail_stop = active_position["highest_bid_seen"] - 1.5 * active_position["entry_atr"]
                if close_p < trail_stop:
                    exit_triggered = True
                    exit_price = trail_stop - DEFAULT_SLIPPAGE
                    
            elif policy_name == "TP 1.5 ATR / SL 1 ATR":
                sl_price = active_position["entry_price"] - 1.0 * active_position["entry_atr"]
                tp_price = active_position["entry_price"] + 1.5 * active_position["entry_atr"]
                
                # Check stops
                if low_p <= sl_price:
                    exit_triggered = True
                    exit_price = sl_price - DEFAULT_SLIPPAGE
                elif high_p >= tp_price:
                    exit_triggered = True
                    exit_price = tp_price - DEFAULT_SLIPPAGE
                    
            elif policy_name == "TP 2 ATR / SL 1 ATR":
                sl_price = active_position["entry_price"] - 1.0 * active_position["entry_atr"]
                tp_price = active_position["entry_price"] + 2.0 * active_position["entry_atr"]
                
                if low_p <= sl_price:
                    exit_triggered = True
                    exit_price = sl_price - DEFAULT_SLIPPAGE
                elif high_p >= tp_price:
                    exit_triggered = True
                    exit_price = tp_price - DEFAULT_SLIPPAGE
                    
            if exit_triggered:
                active_position["end_idx"] = i
                active_position["end_time"] = time_oos[i]
                active_position["exit_price"] = exit_price
                
                pnl = (active_position["exit_price"] - active_position["entry_price"]) * 10.0
                active_position["pnl"] = pnl
                active_position["duration"] = i - active_position["start_idx"]
                
                trades.append(active_position)
                active_position = None
                
    # Close any lingering position at the end of the split
    if active_position is not None:
        end_i = n_oos - 1
        active_position["end_idx"] = end_i
        active_position["end_time"] = time_oos[end_i]
        active_position["exit_price"] = close_oos[end_i] - DEFAULT_SLIPPAGE
        pnl = (active_position["exit_price"] - active_position["entry_price"]) * 10.0
        active_position["pnl"] = pnl
        active_position["duration"] = max(1, end_i - active_position["start_idx"])
        trades.append(active_position)
        
    return trades

def main():
    logger.info("Initializing event holding policy backtests...")
    
    # 1. Load data and models
    symbol = "XAUUSD"
    raw_path = "data/raw/dukascopy"
    processed_path = "data/processed"
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2026, 6, 30)
    
    downloader = DukascopyDownloader(symbol=symbol, raw_dir=raw_path)
    df_raw = downloader.download_range(start_date, end_date)
    
    cleaner = DataCleaner()
    df_cleaned, _ = cleaner.clean_m1_data(df_raw)
    
    tf_generator = TimeframeGenerator(processed_dir=processed_path)
    tf_generator.aggregate_and_save(df_cleaned, symbol=symbol)
    
    df_features = compute_all_features(df_cleaned)
    
    labels_df = get_triple_barrier_labels(
        df_features, 
        atr_col="atr_14", 
        pt_mult=2.0, 
        sl_mult=2.0, 
        max_holding_bars=20
    )
    weights = get_sample_weights(labels_df)
    
    raw_labels = labels_df["label"].values
    y = np.zeros(len(labels_df), dtype=int)
    y[raw_labels == 1.0] = 1
    y[raw_labels == -1.0] = 2
    
    meta_cols = ["timestamp", "open", "high", "low", "close", "volume", "spread", "real_volume"]
    features_cols = [c for c in df_features.columns if c not in meta_cols]
    X = df_features[features_cols].copy()
    
    df_features["timestamp"] = pd.to_datetime(df_features["timestamp"])
    train_mask = (df_features["timestamp"] >= "2024-01-01") & (df_features["timestamp"] <= "2025-12-31 23:59:59")
    val_mask = (df_features["timestamp"] >= "2026-01-01") & (df_features["timestamp"] <= "2026-03-31 23:59:59")
    oos_mask = (df_features["timestamp"] >= "2026-04-01") & (df_features["timestamp"] <= "2026-06-30 23:59:59")
    
    fit_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    oos_idx = np.where(oos_mask)[0]
    
    regime_model = RegimeDetector(n_regimes=3)
    regime_model.fit(df_features.iloc[fit_idx])
    
    entry_model = EntryModel()
    entry_model.fit_and_calibrate(
        X_train=X.iloc[fit_idx],
        y_train=y[fit_idx],
        sample_weight=weights[fit_idx],
        X_calib=X.iloc[val_idx],
        y_calib=y[val_idx]
    )
    
    # Extract prediction outputs
    oos_entry_preds = entry_model.predict(X.iloc[oos_idx])
    oos_entry_probs = entry_model.predict_proba(X.iloc[oos_idx])
    max_probs = np.max(oos_entry_probs, axis=1)
    
    df_all_regime = df_features[regime_model.feature_cols].copy().fillna(0.0)
    state_preds_all = regime_model.hmm.predict(df_all_regime)
    regime_labels_all = np.array([regime_model.state_map.get(idx, "unknown") for idx in state_preds_all])
    regime_labels_oos = regime_labels_all[oos_idx]
    
    # Extract arrays for the simulator loop
    close_oos = df_features.iloc[oos_idx]["close"].values
    high_oos = df_features.iloc[oos_idx]["high"].values
    low_oos = df_features.iloc[oos_idx]["low"].values
    ema21_oos = df_features.iloc[oos_idx]["ema_21"].values
    ema55_oos = df_features.iloc[oos_idx]["ema_55"].values
    atr_oos = df_features.iloc[oos_idx]["atr_14"].values
    time_oos = df_features.iloc[oos_idx]["timestamp"].values
    volume_oos = df_features.iloc[oos_idx]["volume"].values
    
    policies = [
        "EMA21 exit",
        "EMA55 exit",
        "ATR trailing stop",
        "TP 1.5 ATR / SL 1 ATR",
        "TP 2 ATR / SL 1 ATR"
    ]
    thresholds = [0.99, 0.90, 0.80, 0.70, 0.60]
    
    reports_md = []
    
    for thresh in thresholds:
        section = f"## 📊 Confidence Threshold >= {thresh:.0%}\n\n"
        section += "| Exit Policy | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD | Avg Dur (min) | Avg Raw Signals/Trade |\n"
        section += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        
        for pol in policies:
            trades = simulate_policy(
                pol, thresh, oos_idx, max_probs, oos_entry_preds,
                close_oos, high_oos, low_oos, ema21_oos, ema55_oos, atr_oos, regime_labels_oos, time_oos, volume_oos
            )
            n_trades = len(trades)
            if n_trades == 0:
                section += f"| {pol} | 0 | N/A | N/A | $0.00 | 0.00% | N/A | N/A |\n"
                continue
                
            wins = [t for t in trades if t["pnl"] > 0]
            losses = [t for t in trades if t["pnl"] <= 0]
            
            win_rate = len(wins) / n_trades
            avg_dur = np.mean([t["duration"] for t in trades])
            avg_signals = np.mean([t["signals_during_trade"] for t in trades])
            
            total_gains = sum([t["pnl"] for t in wins])
            total_losses = abs(sum([t["pnl"] for t in losses]))
            profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")
            
            pnls = [t["pnl"] for t in trades]
            cost_adjusted_pnl = sum(pnls)
            max_dd = calculate_drawdown(pnls)
            
            section += f"| {pol} | {n_trades} | {win_rate:.2%} | {profit_factor:.2f} | ${cost_adjusted_pnl:.2f} | {max_dd:.2%} | {avg_dur:.1f} | {avg_signals:.2f} |\n"
            
        reports_md.append(section)
        
    report_content = f"""# High Confidence BUY Signal Holding Policies Evaluation Report

This report evaluates performance metrics of high-confidence BUY predictions under five different holding/exit policies. 

## Simulation Controls
- **One Active Trade Limit**: Only one position can be open at any time.
- **No Active Re-entry**: Signal triggers while a position is open are locked out.
- **Execution Cost Inclusions**: BUY entries are executed at Ask (`Close + Spread + Slippage`), and BUY exits are executed at Bid (`Close/Stop - Slippage`).
- **Spread & Slippage Constants**: Spread = 15 points (0.15), Slippage = 5 points (0.05).

---

{"---".join(reports_md)}
"""
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/FIXED_HIGH_CONFIDENCE_BUY_EVENTS_REPORT.md", "w") as f:
        f.write(report_content)
        
    logger.info("Fixed event-holding policy backtest report written successfully!")

if __name__ == "__main__":
    main()
