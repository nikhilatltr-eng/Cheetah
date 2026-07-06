import logging
import time
import datetime
from shared_state import SharedState

logger = logging.getLogger("cheetah_scaler")

class CapitalScaler:
    def __init__(self, shared_state: SharedState, min_trades_per_stage: int = 20, 
                 min_days_per_stage: int = 14, max_drawdown_limit: float = 0.04, 
                 hard_risk_pct_ceiling: float = 0.015):
        """
        Manages the staged capital allocation scale-up schedule.
        Prevents premature scaling and reverts sizing when drawdowns occur.
        """
        self.state = shared_state
        self.min_trades = min_trades_per_stage
        self.min_days = min_days_per_stage
        self.reversion_drawdown = max_drawdown_limit
        self.ceiling = hard_risk_pct_ceiling
        
        # Scaling Stage configurations (stage number -> risk_pct)
        self.stage_config = {
            1: 0.005,    # 0.5% risk (Minimum lot stage)
            2: 0.0075,   # 0.75% risk (25% increment step-up)
            3: 0.010,    # 1.0% risk
            4: 0.0125,   # 1.25% risk
            5: 0.015     # 1.5% risk (Ceiling limit)
        }
        
        self._load_scaler_state()

    def _load_scaler_state(self):
        """Loads or sets default scaler parameters from SharedState."""
        self.current_stage = self.state.get("scaler_stage", 1)
        self.stage_start_time = self.state.get("scaler_stage_start_time", time.time())
        self.stage_start_equity = self.state.get("scaler_stage_start_equity", 10000.0)
        self.stage_peak_equity = self.state.get("scaler_stage_peak_equity", 10000.0)

    def _save_scaler_state(self):
        """Persists parameters to SQLite."""
        self.state.set("scaler_stage", self.current_stage)
        self.state.set("scaler_stage_start_time", self.stage_start_time)
        self.state.set("scaler_stage_start_equity", self.stage_start_equity)
        self.state.set("scaler_stage_peak_equity", self.stage_peak_equity)
        
        # Publish active risk percentage to be read by position_sizer
        active_risk = self.stage_config.get(self.current_stage, 0.005)
        # Enforce hard ceiling safety rule
        active_risk = min(active_risk, self.ceiling)
        self.state.set("risk_pct", active_risk)

    def process_closed_trades(self, current_equity: float, trades_completed_at_stage: list):
        """
        Updates peak equity tracking, checks for drawdown reversions,
        and evaluates stage advancement gates.
        """
        # Load latest values
        self._load_scaler_state()
        
        # 1. Update Peak Equity and calculate Drawdown
        self.stage_peak_equity = max(self.stage_peak_equity, current_equity)
        
        # Calculate drawdown relative to the stage peak
        drawdown = (self.stage_peak_equity - current_equity) / self.stage_peak_equity
        
        logger.info(
            f"CapitalScaler: Stage {self.current_stage} | "
            f"Current Equity: ${current_equity:.2f} | "
            f"Stage Peak: ${self.stage_peak_equity:.2f} | Drawdown: {drawdown:.2%}"
        )
        
        # 2. Check Reversion (Drawdown breach at current stage)
        if drawdown >= self.reversion_drawdown:
            if self.current_stage > 1:
                old_stage = self.current_stage
                self.current_stage -= 1
                self.stage_start_time = time.time()
                self.stage_start_equity = current_equity
                self.stage_peak_equity = current_equity
                self._save_scaler_state()
                logger.error(
                    f"CapitalScaler: Drawdown breach ({drawdown:.2%} >= {self.reversion_drawdown:.2%}). "
                    f"Reverting capital stage: {old_stage} -> {self.current_stage}."
                )
                return
                
        # 3. Check Stage Advancement Gates
        n_trades = len(trades_completed_at_stage)
        days_passed = (time.time() - self.stage_start_time) / 86400.0
        
        logger.info(
            f"CapitalScaler Gates Check: Trades={n_trades}/{self.min_trades} | "
            f"Days={days_passed:.1f}/{self.min_days} | "
            f"Drawdown Limit={drawdown:.2%}/{self.reversion_drawdown:.2%}"
        )
        
        # Gates: min trades met AND min days met AND drawdown is healthy (< 50% of reversion limit)
        if n_trades >= self.min_trades and days_passed >= self.min_days and drawdown < (self.reversion_drawdown * 0.5):
            max_stages = max(self.stage_config.keys())
            if self.current_stage < max_stages:
                old_stage = self.current_stage
                self.current_stage += 1
                self.stage_start_time = time.time()
                self.stage_start_equity = current_equity
                self.stage_peak_equity = current_equity
                self._save_scaler_state()
                logger.warning(
                    f"CapitalScaler: ALL GATES PASSED. Advancing capital stage: {old_stage} -> {self.current_stage}."
                )
            else:
                logger.info("CapitalScaler: Already at maximum configured stage.")
        else:
            logger.info("CapitalScaler: Scaling gates not fully satisfied yet. Stage locked.")
            self._save_scaler_state()
