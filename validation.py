import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)

class PurgedWalkForwardCV:
    def __init__(self, n_folds: int = 5, embargo_bars: int = 10):
        """
        Implements expanding walk-forward cross validation with purging and embargo
        to prevent information leakage in overlapping financial label windows.
        
        Parameters:
            n_folds (int): Number of folds for the split.
            embargo_bars (int): Number of bars to clear after the train fold end
                                (embargo window to prevent leakage from train to test).
        """
        self.n_folds = n_folds
        self.embargo_bars = embargo_bars

    def split(self, df: pd.DataFrame, labels_df: pd.DataFrame):
        """
        Generates purged and embargoed train/test split indices.
        
        Yields:
            train_indices (list of int): Purged train indices.
            test_indices (list of int): Test indices.
        """
        n = len(df)
        if n < 10:
            yield list(range(n)), list(range(n))
            return
            
        segment_size = n // (self.n_folds + 1)
        
        for k in range(1, self.n_folds + 1):
            train_end_idx = k * segment_size
            test_start_idx = train_end_idx
            test_end_idx = min((k + 1) * segment_size, n)
            
            train_indices = list(range(0, train_end_idx))
            test_indices = list(range(test_start_idx, test_end_idx))
            
            # Retrieve test start time and embargo threshold time
            test_start_time = df.loc[test_start_idx, "timestamp"]
            embargo_idx = max(0, test_start_idx - self.embargo_bars)
            embargo_time = df.loc[embargo_idx, "timestamp"]
            
            purged_train_indices = []
            for idx in train_indices:
                label = labels_df.loc[idx, "label"]
                exit_time = labels_df.loc[idx, "exit_time"]
                
                # If label is NaN (warmup), skip it
                if pd.isna(label):
                    continue
                    
                # If the exit time overlaps into the embargo window or test window, purge it
                if pd.notna(exit_time) and exit_time >= embargo_time:
                    continue
                    
                purged_train_indices.append(idx)
                
            yield purged_train_indices, test_indices

def compute_fold_performance(df: pd.DataFrame, y_pred: np.ndarray, test_indices: list) -> dict:
    """
    Computes hypothetical performance metrics for a fold's test predictions.
    Assumes prediction:
      - 1 (LONG / BUY): long position held for 1 bar
      - 2 (SHORT / SELL): short position held for 1 bar
      - 0 (HOLD / NO-TRADE): flat position
    """
    test_df = df.iloc[test_indices].copy()
    
    # Map predictions from LightGBM classes [0, 1, 2] -> signal directions [0, 1, -1]
    # Class 0 -> 0 (no trade)
    # Class 1 -> 1 (long)
    # Class 2 -> -1 (short)
    signal = np.zeros(len(y_pred))
    signal[y_pred == 1] = 1.0
    signal[y_pred == 2] = -1.0
    
    test_df["pred_signal"] = signal
    
    # Standard log returns of close
    test_df["gold_ret"] = test_df["close"].pct_change().fillna(0.0)
    
    # Shift predictions by 1 to represent entering trade at close of bar t and realizing returns at t+1
    test_df["strategy_ret"] = test_df["pred_signal"].shift(1).fillna(0.0) * test_df["gold_ret"]
    
    # Calculate Sharpe (annualized, assuming daily timescale as a proxy)
    mean_ret = test_df["strategy_ret"].mean()
    std_ret = test_df["strategy_ret"].std()
    
    if std_ret > 0:
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
    else:
        sharpe = 0.0
        
    # Calculate Max Drawdown
    cum_ret = (1.0 + test_df["strategy_ret"]).cumprod()
    running_max = cum_ret.cummax()
    # Handle initial condition
    running_max = np.where(running_max == 0, 1.0, running_max)
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min() if len(drawdown) > 0 else 0.0
    
    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd)
    }

def generate_fold_report(y_true: np.ndarray, y_pred: np.ndarray, perf_dict: dict) -> dict:
    """Generates detailed precision/recall stats per class for a fold."""
    # Classes: 0 (No-trade), 1 (Long), 2 (Short)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    
    report = {
        "no_trade": {"precision": float(precision[0]), "recall": float(recall[0]), "support": int(support[0])},
        "long": {"precision": float(precision[1]), "recall": float(recall[1]), "support": int(support[1])},
        "short": {"precision": float(precision[2]), "recall": float(recall[2]), "support": int(support[2])},
        "sharpe": perf_dict["sharpe"],
        "max_drawdown": perf_dict["max_drawdown"]
    }
    return report
