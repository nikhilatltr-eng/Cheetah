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
from validation.significance_filter import filter_table_by_significance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sell_side_holding_report")

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

def simulate_sell_policy(policy_name, threshold, spread, slippage, session_name, oos_idx, max_probs, oos_entry_preds, close_oos, high_oos, low_oos, ema21_oos, ema55_oos, atr_oos, regime_labels_oos, time_oos, volume_oos):
    """
    Simulates position holding for a SELL policy.
    One open position only. No re-entry while active. Only enter on active market bars.
    SELL entry executes at Bid (Close - Spread - Slippage).
    SELL exit executes at Ask (Close/Stop + Slippage).
    """
    trades = []
    active_position = None
    n_oos = len(oos_idx)
    
    for i in range(n_oos):
        conf = max_probs[i]
        pred = oos_entry_preds[i]
        vol = volume_oos[i]
        dt = pd.to_datetime(time_oos[i])
        
        # Pred 2 is SELL
        is_signal = (pred == 2) and (conf >= threshold) and (vol > 0.0)
        
        close_p = close_oos[i]
        high_p = high_oos[i]
        low_p = low_oos[i]
        ema21 = ema21_oos[i]
        ema55 = ema55_oos[i]
        atr = atr_oos[i]
        
        if active_position is None:
            if is_signal and check_session(dt, session_name):
                # Enter Short at Bid
                active_position = {
                    "start_idx": i,
                    "start_time": time_oos[i],
                    "entry_price": close_p - spread - slippage,
                    "entry_atr": atr,
                    "lowest_ask_seen": close_p,
                    "signals_during_trade": 1
                }
        else:
            if (pred == 2) and (conf >= threshold):
                active_position["signals_during_trade"] += 1
                
            active_position["lowest_ask_seen"] = min(active_position["lowest_ask_seen"], close_p)
            
            exit_triggered = False
            exit_price = close_p + slippage  # Default Ask exit price
            
            if policy_name == "EMA21 exit":
                if close_p > ema21:
                    exit_triggered = True
            elif policy_name == "EMA55 exit":
                if close_p > ema55:
                    exit_triggered = True
            elif policy_name == "ATR trailing stop":
                trail_stop = active_position["lowest_ask_seen"] + 1.5 * active_position["entry_atr"]
                if close_p > trail_stop:
                    exit_triggered = True
                    exit_price = trail_stop + slippage
            elif policy_name == "TP 1.5 ATR / SL 1 ATR":
                sl_price = active_position["entry_price"] + 1.0 * active_position["entry_atr"]
                tp_price = active_position["entry_price"] - 1.5 * active_position["entry_atr"]
                
                if high_p >= sl_price:
                    exit_triggered = True
                    exit_price = sl_price + slippage
                elif low_p <= tp_price:
                    exit_triggered = True
                    exit_price = tp_price + slippage
            elif policy_name == "TP 2 ATR / SL 1 ATR":
                sl_price = active_position["entry_price"] + 1.0 * active_position["entry_atr"]
                tp_price = active_position["entry_price"] - 2.0 * active_position["entry_atr"]
                
                if high_p >= sl_price:
                    exit_triggered = True
                    exit_price = sl_price + slippage
                elif low_p <= tp_price:
                    exit_triggered = True
                    exit_price = tp_price + slippage
                    
            if exit_triggered:
                active_position["end_idx"] = i
                active_position["end_time"] = time_oos[i]
                active_position["exit_price"] = exit_price
                
                # PnL for Short trade is (Entry - Exit) * 10.0
                pnl = (active_position["entry_price"] - active_position["exit_price"]) * 10.0
                active_position["pnl"] = pnl
                active_position["duration"] = i - active_position["start_idx"]
                
                trades.append(active_position)
                active_position = None
                
    if active_position is not None:
        end_i = n_oos - 1
        active_position["end_idx"] = end_i
        active_position["end_time"] = time_oos[end_i]
        active_position["exit_price"] = close_oos[end_i] + slippage
        pnl = (active_position["entry_price"] - active_position["exit_price"]) * 10.0
        active_position["pnl"] = pnl
        active_position["duration"] = max(1, end_i - active_position["start_idx"])
        trades.append(active_position)
        
    return trades

def generate_sell_report_data(config_path="config.yaml"):
    logger.info("Running SELL-side holding policy simulations...")
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
    labels_df = get_triple_barrier_labels(df_features, atr_col="atr_14", pt_mult=2.0, sl_mult=2.0, max_holding_bars=20)
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
    max_probs = oos_entry_probs[:, 2]  # Probability of SELL
    
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
    
    return (oos_idx, max_probs, oos_entry_preds, close_oos, high_oos, low_oos, ema21_oos, ema55_oos, atr_oos, regime_labels_oos, time_oos, volume_oos)

if __name__ == "__main__":
    generate_sell_report_data()
