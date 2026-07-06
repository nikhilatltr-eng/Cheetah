import os
import sys
import time
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from entry_model import EntryModel
from tft_entry_model import LSTMEntryModel

def test_lstm_vs_gbm_ablation_performance():
    """
    Fits both LightGBM (EntryModel) and PyTorch LSTM (LSTMEntryModel) on identical folds
    and asserts that a performance comparison report is written to disk with real numbers.
    """
    # 1. Generate synthetic dataset
    np.random.seed(42)
    n_samples = 150
    n_features = 8
    
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(np.random.normal(0, 1, (n_samples, n_features)), columns=feature_names)
    
    # 3-class targets: 0: neutral, 1: buy, 2: sell
    # Map raw targets with some correlation to features to allow learning
    raw_signal = X["feat_0"].values * 0.5 + X["feat_1"].values * 0.3
    y = np.zeros(n_samples, dtype=int)
    y[raw_signal > 0.4] = 1
    y[raw_signal < -0.4] = 2
    
    # Split into Train (100 samples) and Test (50 samples)
    X_train, X_test = X.iloc[:100], X.iloc[100:]
    y_train, y_test = y[:100], y[100:]
    
    # 2. Benchmark Baseline LightGBM Model
    t0 = time.perf_counter()
    gbm_model = EntryModel()
    # LightGBM requires calibration split, we simulate fitting
    gbm_model.fit_and_calibrate(X_train.iloc[:80], y_train[:80], None, X_train.iloc[80:], y_train[80:])
    gbm_fit_time = time.perf_counter() - t0
    
    gbm_preds = gbm_model.predict(X_test)
    gbm_acc = np.mean(gbm_preds == y_test)
    
    # 3. Benchmark PyTorch LSTM Sequence Model
    t0 = time.perf_counter()
    lstm_model = LSTMEntryModel(seq_len=5, epochs=3, batch_size=16)
    lstm_model.fit(X_train, y_train)
    lstm_fit_time = time.perf_counter() - t0
    
    lstm_preds = lstm_model.predict(X_test)
    # The LSTM outputs padded zeros for the first W-1 steps during inference
    # So we compare only starting from seq_len - 1 to be fair
    valid_idx = lstm_model.seq_len - 1
    lstm_acc = np.mean(lstm_preds[valid_idx:] == y_test[valid_idx:])
    
    # 4. Generate Ablation Report text
    report_path = "tests/ablation_report.txt"
    report_content = f"""======================================================
ABLATION REPORT: LIGHTGBM VS PYTORCH LSTM ENTRY MODEL
Generated: {pd.Timestamp.now().isoformat()}
======================================================
Dataset Parameters:
  - Total Samples: {n_samples}
  - Train Samples: 100
  - Test Samples: 50
  - Number of Features: {n_features}

LightGBM Entry Model (Champion):
  - Fit Time: {gbm_fit_time*1000:.2f} ms
  - Out-of-Sample Accuracy: {gbm_acc:.2%}

PyTorch LSTM Sequence Model (Challenger):
  - Fit Time: {lstm_fit_time*1000:.2f} ms
  - Out-of-Sample Accuracy (Inference-Aligned): {lstm_acc:.2%}

Conclusion:
  {"PROMOTED" if lstm_acc > gbm_acc else "REJECTED (LightGBM baseline remains champion)"}
======================================================
"""
    
    # Write report to disk
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print("\n" + report_content)
    
    # Assertions
    assert os.path.exists(report_path)
    assert len(gbm_preds) == len(y_test)
    assert len(lstm_preds) == len(y_test)
    assert gbm_fit_time > 0
    assert lstm_fit_time > 0
