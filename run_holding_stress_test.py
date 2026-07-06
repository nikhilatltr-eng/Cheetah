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
logger = logging.getLogger("holding_stress_test")

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

def check_session(dt, session_name):
    if session_name == "London only":
        return 8 <= dt.hour < 16
    elif session_name == "New York only":
        return 13 <= dt.hour < 21
    elif session_name == "London + NY":
        return 8 <= dt.hour < 21
    else:  # All sessions
        return True

def simulate_policy_stress(policy_name, threshold, spread, slippage, session_name, oos_idx, max_probs, oos_entry_preds, close_oos, high_oos, low_oos, ema21_oos, ema55_oos, atr_oos, regime_labels_oos, time_oos, volume_oos):
    trades = []
    active_position = None
    n_oos = len(oos_idx)
    
    for i in range(n_oos):
        conf = max_probs[i]
        pred = oos_entry_preds[i]
        vol = volume_oos[i]
        dt = pd.to_datetime(time_oos[i])
        
        is_signal = (pred == 1) and (conf >= threshold) and (vol > 0.0)
        
        close_p = close_oos[i]
        high_p = high_oos[i]
        low_p = low_oos[i]
        ema21 = ema21_oos[i]
        ema55 = ema55_oos[i]
        atr = atr_oos[i]
        
        if active_position is None:
            if is_signal and check_session(dt, session_name):
                # Enter Long at Ask
                active_position = {
                    "start_idx": i,
                    "start_time": time_oos[i],
                    "entry_price": close_p + spread + slippage,
                    "entry_atr": atr,
                    "highest_bid_seen": close_p,
                    "signals_during_trade": 1
                }
        else:
            if (pred == 1) and (conf >= threshold):
                active_position["signals_during_trade"] += 1
                
            active_position["highest_bid_seen"] = max(active_position["highest_bid_seen"], close_p)
            
            exit_triggered = False
            exit_price = close_p - slippage
            
            if policy_name == "EMA21 exit":
                if close_p < ema21:
                    exit_triggered = True
            elif policy_name == "ATR trailing stop":
                trail_stop = active_position["highest_bid_seen"] - 1.5 * active_position["entry_atr"]
                if close_p < trail_stop:
                    exit_triggered = True
                    exit_price = trail_stop - slippage
                    
            if exit_triggered:
                active_position["end_idx"] = i
                active_position["end_time"] = time_oos[i]
                active_position["exit_price"] = exit_price
                
                pnl = (active_position["exit_price"] - active_position["entry_price"]) * 10.0
                active_position["pnl"] = pnl
                active_position["duration"] = i - active_position["start_idx"]
                
                trades.append(active_position)
                active_position = None
                
    if active_position is not None:
        end_i = n_oos - 1
        active_position["end_idx"] = end_i
        active_position["end_time"] = time_oos[end_i]
        active_position["exit_price"] = close_oos[end_i] - slippage
        pnl = (active_position["exit_price"] - active_position["entry_price"]) * 10.0
        active_position["pnl"] = pnl
        active_position["duration"] = max(1, end_i - active_position["start_idx"])
        trades.append(active_position)
        
    return trades

def main():
    logger.info("Initializing comprehensive holding policy stress backtests...")
    
    # Load data
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
    
    oos_entry_preds = entry_model.predict(X.iloc[oos_idx])
    oos_entry_probs = entry_model.predict_proba(X.iloc[oos_idx])
    max_probs = np.max(oos_entry_probs, axis=1)
    
    df_all_regime = df_features[regime_model.feature_cols].copy().fillna(0.0)
    state_preds_all = regime_model.hmm.predict(df_all_regime)
    regime_labels_all = np.array([regime_model.state_map.get(idx, "unknown") for idx in state_preds_all])
    regime_labels_oos = regime_labels_all[oos_idx]
    
    close_oos = df_features.iloc[oos_idx]["close"].values
    high_oos = df_features.iloc[oos_idx]["high"].values
    low_oos = df_features.iloc[oos_idx]["low"].values
    ema21_oos = df_features.iloc[oos_idx]["ema_21"].values
    ema55_oos = df_features.iloc[oos_idx]["ema_55"].values
    atr_oos = df_features.iloc[oos_idx]["atr_14"].values
    time_oos = df_features.iloc[oos_idx]["timestamp"].values
    volume_oos = df_features.iloc[oos_idx]["volume"].values
    
    # Stress parameters
    cost_cases = [
        {"name": "1x cost", "spread": 0.15, "slippage": 0.05},
        {"name": "2x cost", "spread": 0.30, "slippage": 0.10},
        {"name": "3x cost", "spread": 0.45, "slippage": 0.15},
        {"name": "0.40 Spread Equivalent", "spread": 0.40, "slippage": 0.05}
    ]
    
    session_cases = ["London only", "New York only", "London + NY", "All sessions"]
    strategies = ["EMA21 exit", "ATR trailing stop"]
    thresholds = [0.70, 0.60]
    
    reports_md = []
    
    for thresh in thresholds:
        section = f"## 📊 Confidence Threshold >= {thresh:.0%}\n\n"
        
        # 1. Cost stress tables for each strategy
        for strat in strategies:
            section += f"### 🛡️ Cost Stress Test: {strat}\n\n"
            section += "| Cost Level | Spread | Slippage | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |\n"
            section += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
            
            for cost in cost_cases:
                trades = simulate_policy_stress(
                    strat, thresh, cost["spread"], cost["slippage"], "All sessions",
                    oos_idx, max_probs, oos_entry_preds, close_oos, high_oos, low_oos,
                    ema21_oos, ema55_oos, atr_oos, regime_labels_oos, time_oos, volume_oos
                )
                n_trades = len(trades)
                if n_trades == 0:
                    section += f"| {cost['name']} | {cost['spread']:.2f} | {cost['slippage']:.2f} | 0 | N/A | N/A | $0.00 | 0.00% |\n"
                    continue
                wins = [t for t in trades if t["pnl"] > 0]
                losses = [t for t in trades if t["pnl"] <= 0]
                win_rate = len(wins) / n_trades
                total_gains = sum([t["pnl"] for t in wins])
                total_losses = abs(sum([t["pnl"] for t in losses]))
                profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")
                pnls = [t["pnl"] for t in trades]
                pnl_sum = sum(pnls)
                max_dd = calculate_drawdown(pnls)
                
                section += f"| {cost['name']} | {cost['spread']:.2f} | {cost['slippage']:.2f} | {n_trades} | {win_rate:.2%} | {profit_factor:.2f} | ${pnl_sum:.2f} | {max_dd:.2%} |\n"
            section += "\n"
            
        # 2. Session filter tables for each strategy under base 1x cost
        for strat in strategies:
            section += f"### ⏰ Trading Sessions Stress Test: {strat} (1x Cost)\n\n"
            section += "| Session | Entry Hours (UTC) | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |\n"
            section += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
            
            for sess in session_cases:
                hours_desc = "08:00 - 16:00" if sess == "London only" else ("13:00 - 21:00" if sess == "New York only" else ("08:00 - 21:00" if sess == "London + NY" else "24h"))
                trades = simulate_policy_stress(
                    strat, thresh, 0.15, 0.05, sess,
                    oos_idx, max_probs, oos_entry_preds, close_oos, high_oos, low_oos,
                    ema21_oos, ema55_oos, atr_oos, regime_labels_oos, time_oos, volume_oos
                )
                n_trades = len(trades)
                if n_trades == 0:
                    section += f"| {sess} | {hours_desc} | 0 | N/A | N/A | $0.00 | 0.00% |\n"
                    continue
                wins = [t for t in trades if t["pnl"] > 0]
                losses = [t for t in trades if t["pnl"] <= 0]
                win_rate = len(wins) / n_trades
                total_gains = sum([t["pnl"] for t in wins])
                total_losses = abs(sum([t["pnl"] for t in losses]))
                profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")
                pnls = [t["pnl"] for t in trades]
                pnl_sum = sum(pnls)
                max_dd = calculate_drawdown(pnls)
                
                section += f"| {sess} | {hours_desc} | {n_trades} | {win_rate:.2%} | {profit_factor:.2f} | ${pnl_sum:.2f} | {max_dd:.2%} |\n"
            section += "\n"
            
        # 3. Monthly breakdown for each strategy under base 1x cost
        for strat in strategies:
            section += f"### 📅 Monthly Performance Breakdown: {strat} (1x Cost, All Sessions)\n\n"
            section += "| Month | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |\n"
            section += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            
            trades = simulate_policy_stress(
                strat, thresh, 0.15, 0.05, "All sessions",
                oos_idx, max_probs, oos_entry_preds, close_oos, high_oos, low_oos,
                ema21_oos, ema55_oos, atr_oos, regime_labels_oos, time_oos, volume_oos
            )
            
            if len(trades) > 0:
                df_tr = pd.DataFrame(trades)
                df_tr["month"] = pd.to_datetime(df_tr["start_time"]).dt.strftime("%Y-%m")
                months_sorted = sorted(df_tr["month"].unique())
                
                for m in months_sorted:
                    m_tr = df_tr[df_tr["month"] == m]
                    n_m = len(m_tr)
                    m_wins = m_tr[m_tr["pnl"] > 0]
                    m_losses = m_tr[m_tr["pnl"] <= 0]
                    w_r = len(m_wins) / n_m
                    gains = sum(m_wins["pnl"])
                    loss_sum = abs(sum(m_losses["pnl"]))
                    pf = gains / loss_sum if loss_sum > 0 else float("inf")
                    pnl_val = sum(m_tr["pnl"])
                    dd_val = calculate_drawdown(list(m_tr["pnl"]))
                    
                    section += f"| {m} | {n_m} | {w_r:.2%} | {pf:.2f} | ${pnl_val:.2f} | {dd_val:.2%} |\n"
            else:
                section += "| N/A | 0 | N/A | N/A | $0.00 | 0.00% |\n"
            section += "\n"
            
        reports_md.append(section)
        
    report_content = f"""# High Confidence BUY Signal Holding Policies Stress Test Report

This report evaluates cost resilience, trading session filters, and monthly performance breakdowns of our best-performing BUY-only holding strategies on the out-of-sample (OOS) test partition.

## Strategy Target Configurations
1. **EMA21 exit**: Long position closed immediately when M5 price closes below EMA21.
2. **ATR trailing stop**: Trailing stop set at `Highest Bid observed - 1.5 * ATR_14`.

---

{"---".join(reports_md)}
"""
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/FIXED_HIGH_CONFIDENCE_BUY_EVENTS_REPORT.md", "a") as f:
        f.write("\n\n" + report_content)
        
    logger.info("Stress test report appended successfully to reports/FIXED_HIGH_CONFIDENCE_BUY_EVENTS_REPORT.md!")

if __name__ == "__main__":
    main()
