import datetime
import pandas as pd
import numpy as np
from features import compute_all_features
from labeling import get_triple_barrier_labels, get_sample_weights
from entry_model import EntryModel
from regime_model import RegimeDetector
from data.dukascopy_downloader import DukascopyDownloader
from data.data_cleaner import DataCleaner

symbol = "XAUUSD"
raw_path = "data/raw/dukascopy"
start_date = datetime.date(2024, 1, 1)
end_date = datetime.date(2026, 6, 30)

downloader = DukascopyDownloader(symbol=symbol, raw_dir=raw_path)
df_raw = downloader.download_range(start_date, end_date)
cleaner = DataCleaner()
df_cleaned, _ = cleaner.clean_m1_data(df_raw)

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

meta_cols = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
features_cols = [c for c in df_features.columns if c not in meta_cols]
X = df_features[features_cols].copy()

df_features["timestamp"] = pd.to_datetime(df_features["timestamp"])
train_mask = (df_features["timestamp"] >= "2024-01-01") & (df_features["timestamp"] <= "2025-12-31 23:59:59")
val_mask = (df_features["timestamp"] >= "2026-01-01") & (df_features["timestamp"] <= "2026-03-31 23:59:59")
oos_mask = (df_features["timestamp"] >= "2026-04-01") & (df_features["timestamp"] <= "2026-06-30 23:59:59")

fit_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
oos_idx = np.where(oos_mask)[0]

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

print("Max confidence distribution (all OOS):")
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
counts, _ = np.histogram(max_probs, bins=bins)
for i in range(len(counts)):
    print(f"{bins[i]:.2f}-{bins[i+1]:.2f}: {counts[i]}")
    
# Check BUY only confidence
buy_mask = oos_entry_preds == 1
buy_probs = max_probs[buy_mask]
print("\nBUY confidence distribution:")
counts_buy, _ = np.histogram(buy_probs, bins=bins)
for i in range(len(counts_buy)):
    print(f"{bins[i]:.2f}-{bins[i+1]:.2f}: {counts_buy[i]}")
