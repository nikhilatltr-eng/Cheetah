import logging
import numpy as np
import scipy.stats as stats
import pandas as pd
from shared_state import SharedState

logger = logging.getLogger("cheetah_drift")

class PerformanceDriftMonitor:
    def __init__(self, shared_state: SharedState, expected_win_rate: float = 0.52, 
                 expected_r_mean: float = 0.20, expected_r_std: float = 1.0, 
                 min_trades_to_test: int = 10, p_value_alpha: float = 0.05):
        """
        Monitors live trade outcomes to detect statistical performance drift
        from backtested expectations.
        """
        self.state = shared_state
        self.expected_win_rate = expected_win_rate
        self.expected_r_mean = expected_r_mean
        self.expected_r_std = expected_r_std
        self.min_trades = min_trades_to_test
        self.alpha = p_value_alpha
        
        # Build simulated baseline backtest R-multiples distribution for KS-test comparison
        np.random.seed(42)
        self.backtest_r_baseline = np.random.normal(expected_r_mean, expected_r_std, 150)

    def evaluate_drift(self, completed_trades_df: pd.DataFrame) -> dict:
        """
        Performs:
          - Proportion Z-Test: Checks if live win rate is significantly lower than expected.
          - Kolmogorov-Smirnov Test: Checks if live R-multiples distribution deviates from backtest.
        If a breach is verified, triggers is_halted in SharedState.
        """
        n = len(completed_trades_df)
        if n < self.min_trades:
            return {
                "drift_detected": False,
                "n_trades": n,
                "win_rate_p_val": 1.0,
                "r_dist_p_val": 1.0,
                "reason": f"Insufficient trades ({n}/{self.min_trades}) to execute statistical tests."
            }

        # 1. Extract live metrics
        r_multiples = completed_trades_df["r_multiple"].values
        pnls = completed_trades_df["pnl"].values
        
        wins = sum(1 for p in pnls if p > 0.0)
        live_win_rate = wins / n
        
        # 2. Z-test on proportions (Win Rate)
        # H0: live_win_rate >= expected_win_rate
        # H1: live_win_rate < expected_win_rate
        p0 = self.expected_win_rate
        se = np.sqrt(p0 * (1.0 - p0) / n)
        z_stat = (live_win_rate - p0) / se
        # One-tailed p-value
        p_val_win = float(stats.norm.cdf(z_stat))
        
        # 3. Kolmogorov-Smirnov 2-sample test (R-multiples distribution comparison)
        # H0: Live and backtest R-multiples are drawn from the same continuous distribution
        # H1: They are drawn from different distributions
        ks_stat, p_val_r = stats.ks_2samp(r_multiples, self.backtest_r_baseline)
        
        # 4. Evaluate breaches
        drift_detected = False
        reason = "System performing within acceptable statistical limits."
        
        if p_val_win < self.alpha:
            drift_detected = True
            reason = f"Win Rate degradation: Live = {live_win_rate:.2%} vs Expected = {p0:.2%} (Z-test p-value: {p_val_win:.4f} < {self.alpha})"
        elif p_val_r < self.alpha:
            drift_detected = True
            reason = f"R-multiple distribution shift: KS Stat = {ks_stat:.4f} (KS p-value: {p_val_r:.4f} < {self.alpha})"

        if drift_detected:
            logger.error(f"PerformanceDriftMonitor: DRIFT BREACH DETECTED: {reason}")
            # Set SharedState halt flags to lock new entries
            self.state.set("is_halted", True)
            self.state.set("halt_reason", f"Drift Breach: {reason}")
            
        return {
            "drift_detected": drift_detected,
            "n_trades": n,
            "win_rate": float(live_win_rate),
            "win_rate_p_val": float(p_val_win),
            "r_dist_p_val": float(p_val_r),
            "reason": reason
        }
