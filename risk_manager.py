import logging
import datetime

logger = logging.getLogger("cheetah_risk")

class RiskManager:
    def __init__(self, max_daily_loss_pct: float = 0.02, max_positions: int = 3, 
                 circuit_breaker_trades: int = 10, circuit_breaker_min_win_rate: float = 0.35):
        """
        Hard risk control checks running before every execution.
        Includes:
          - Daily Loss limit halt
          - Position cap limits
          - Correlation direction checks (single direction per asset)
          - Rolling win-rate performance circuit breaker
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_positions = max_positions
        self.circuit_breaker_trades = circuit_breaker_trades
        self.circuit_breaker_min_win_rate = circuit_breaker_min_win_rate
        
        self.daily_start_equity = None
        self.last_reset_date = None
        self.is_halted = False
        self.halt_reason = ""

    def reset_daily_equity_if_new_day(self, current_equity: float):
        """Resets the daily starting equity anchor at UTC midnight/new day."""
        today = datetime.datetime.now(datetime.timezone.utc).date()
        if self.last_reset_date is None or today != self.last_reset_date:
            self.daily_start_equity = current_equity
            self.last_reset_date = today
            self.is_halted = False
            self.halt_reason = ""
            logger.info(f"RiskManager: New trading day started. Anchoring daily start equity to ${current_equity:.2f}")

    def evaluate_risk(self, current_equity: float, active_positions: list, trade_history: list, proposed_direction: str) -> tuple:
        """
        Runs all risk checks.
        Returns:
            (bool, str): (is_trade_allowed, block_reason)
        """
        # Ensure daily anchor is active
        self.reset_daily_equity_if_new_day(current_equity)
        
        if self.is_halted:
            return False, f"Trading is currently Halted: {self.halt_reason}"
            
        # 1. Daily Loss Limit Check
        daily_loss_limit = self.daily_start_equity * self.max_daily_loss_pct
        current_loss = self.daily_start_equity - current_equity
        
        if current_loss >= daily_loss_limit:
            self.is_halted = True
            self.halt_reason = f"Daily Loss Limit hit. Loss = ${current_loss:.2f} (Limit: ${daily_loss_limit:.2f})"
            logger.error(f"RiskManager: {self.halt_reason}. Halting all trading activities.")
            return False, self.halt_reason
            
        # 2. Maximum Concurrent Positions Check
        if len(active_positions) >= self.max_positions:
            return False, f"Maximum positions limit reached ({len(active_positions)}/{self.max_positions})."
            
        # 3. Correlation-Aware Exposure Check (Single direction constraint for XAUUSD)
        # proposed_direction is either 'BUY' or 'SELL'
        for pos in active_positions:
            if pos["type"] == proposed_direction:
                return False, f"Exposure Block: Already holding an active {proposed_direction} position on XAUUSD."
                
        # 4. Performance Circuit Breaker
        # Monitor rolling win rate of the last N trades in history
        if len(trade_history) >= self.circuit_breaker_trades:
            recent_trades = trade_history[-self.circuit_breaker_trades:]
            wins = sum(1 for t in recent_trades if t.get("pnl", 0.0) > 0.0)
            win_rate = wins / self.circuit_breaker_trades
            
            if win_rate < self.circuit_breaker_min_win_rate:
                self.is_halted = True
                self.halt_reason = f"Performance Circuit Breaker: Win rate of last {self.circuit_breaker_trades} trades is {win_rate:.2%} (Minimum expected: {self.circuit_breaker_min_win_rate:.2%})"
                logger.error(f"RiskManager: {self.halt_reason}. Trading halted.")
                return False, self.halt_reason
                
        return True, ""
        
    def resume_trading(self):
        """Manually overrides and resets halts for testing or recovery."""
        self.is_halted = False
        self.halt_reason = ""
        logger.info("RiskManager: Trading halts manually cleared and resumed.")
