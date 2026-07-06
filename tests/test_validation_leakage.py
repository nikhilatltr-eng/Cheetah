import os
import sys
import pandas as pd
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation import PurgedWalkForwardCV

def test_validation_leakage_purging():
    """
    Constructs overlapping labels and verifies that the validation splitter
    purges train samples whose exit times overlap with the test fold start index
    (including the embargo window).
    """
    n_bars = 20
    dates = pd.date_range("2026-07-05", periods=n_bars, freq="min", tz="UTC")
    
    df = pd.DataFrame({
        "timestamp": dates,
        "close": [100.0] * n_bars
    })
    
    # We will mock the labels dataframe
    # Suppose test_start_idx = 10.
    # Embargo = 2 bars.
    # Embargo index = 10 - 2 = 8.
    # Embargo time threshold is dates[8].
    #
    # Any training row (0 to 9) whose exit_time is >= dates[8] must be purged.
    labels_df = pd.DataFrame({
        "label": [1.0] * n_bars,
        "exit_time": [
            dates[12],  # Row 0: Exit is dates[12] (>= dates[8]) -> Purged
            dates[10],  # Row 1: Exit is dates[10] (>= dates[8]) -> Purged
            dates[8],   # Row 2: Exit is dates[8]  (>= dates[8]) -> Purged
            dates[7],   # Row 3: Exit is dates[7]  (< dates[8])  -> Keep
            dates[5],   # Row 4: Exit is dates[5]  (< dates[8])  -> Keep
            dates[4],   # Row 5: Exit is dates[4]  (< dates[8])  -> Keep
            dates[6],   # Row 6: Exit is dates[6]  (< dates[8])  -> Keep
            dates[7],   # Row 7: Exit is dates[7]  (< dates[8])  -> Keep
            dates[8],   # Row 8: Exit is dates[8]  (>= dates[8]) -> Purged
            dates[9],   # Row 9: Exit is dates[9]  (>= dates[8]) -> Purged
            dates[15],  # Row 10: Test sample
            dates[15],  # Row 11: Test sample
            dates[15],  # Row 12: Test sample
            dates[15],  # Row 13: Test sample
            dates[15],  # Row 14: Test sample
            dates[15],  # Row 15: Test sample
            dates[15],  # Row 16: Test sample
            dates[15],  # Row 17: Test sample
            dates[15],  # Row 18: Test sample
            dates[15]   # Row 19: Test sample
        ]
    })
    
    # We setup the CV with 1 fold so that the train split boundary falls exactly on index 10.
    # Segment size = N / (K + 1) = 20 / 2 = 10.
    # Fold 1: train_end = 10, test_start = 10, test_end = 20.
    cv = PurgedWalkForwardCV(n_folds=1, embargo_bars=2)
    
    splits = list(cv.split(df, labels_df))
    assert len(splits) == 1
    
    purged_train_idx, test_idx = splits[0]
    
    # Expected test indices are 10 to 19
    assert test_idx == list(range(10, 20))
    
    # Expected kept training indices are 3, 4, 5, 6, 7
    # 0, 1, 2, 8, 9 should be purged!
    assert 0 not in purged_train_idx
    assert 1 not in purged_train_idx
    assert 2 not in purged_train_idx
    assert 8 not in purged_train_idx
    assert 9 not in purged_train_idx
    
    assert 3 in purged_train_idx
    assert 4 in purged_train_idx
    assert 5 in purged_train_idx
    assert 6 in purged_train_idx
    assert 7 in purged_train_idx
    
    assert len(purged_train_idx) == 5
