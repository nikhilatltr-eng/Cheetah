import logging
import numpy as np

logger = logging.getLogger(__name__)

class PositionSizer:
    def __init__(self, risk_pct: float = 0.01, use_kelly: bool = False, kelly_fraction: float = 0.5, 
                 max_kelly_size: float = 0.15, confidence_threshold: float = 0.6, min_lot: float = 0.01, max_lot: float = 10.0):
        """
        Calculates volatility-adjusted and model-confidence-scaled position sizes.
        Includes optional fractional-Kelly scaling and strict lot-size constraints.
        """
        self.risk_pct = risk_pct
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        self.max_kelly_size = max_kelly_size
        self.confidence_threshold = confidence_threshold
        self.min_lot = min_lot
        self.max_lot = max_lot

    def calculate_size(self, equity: float, atr: float, stop_multiple: float, confidence: float, 
                       point_value: float = 100.0, pt_mult: float = 2.0, sl_mult: float = 2.0) -> float:
        """
        Computes position size in lots.
        Formula:
          - Volatility-adjusted size: lot_size = (equity * risk) / (ATR * stop_multiple * point_value)
          - Kelly sizing: scales standard size by fractional Kelly outcome
          - Confidence scaling: down-sizes the lot size if confidence is close to the threshold
        """
        if atr <= 0:
            logger.warning("PositionSizer: Invalid ATR <= 0. Returning minimum lot.")
            return self.min_lot
            
        # 1. Base volatility-adjusted sizing
        risk_amount = equity * self.risk_pct
        stop_loss_distance = atr * stop_multiple
        
        # Sizing formula for standard futures/forex contracts
        base_lots = risk_amount / (stop_loss_distance * point_value + 1e-6)
        
        # 2. Optional Kelly scaling
        # Kelly: f* = (p * (b + 1) - 1) / b
        # where p = win probability (model confidence), b = win/loss ratio (pt_mult / sl_mult)
        if self.use_kelly:
            p = confidence
            b = pt_mult / (sl_mult + 1e-6)
            
            # Compute raw Kelly fraction
            raw_kelly = (p * (b + 1.0) - 1.0) / (b + 1e-6)
            
            # Apply fractional factor (fractional Kelly) and bound it
            fractional_kelly = max(0.0, raw_kelly * self.kelly_fraction)
            fractional_kelly = min(fractional_kelly, self.max_kelly_size)
            
            # Adjust lots: multiply by (fractional_kelly / risk_pct) to match fractional-Kelly capital allocation
            # or treat fractional_kelly as the new risk percentage
            risk_ratio = fractional_kelly / self.risk_pct
            base_lots *= risk_ratio
            logger.debug(f"PositionSizer: Kelly multiplier applied = {risk_ratio:.4f}")
            
        # 3. Confidence scaling
        # If prediction confidence is below the threshold, scale size down linearly
        if confidence < self.confidence_threshold:
            # Slices size down as confidence approaches random guess (0.50)
            confidence_scale = max(0.0, (confidence - 0.50) / (self.confidence_threshold - 0.50 + 1e-6))
            base_lots *= confidence_scale
            logger.info(f"PositionSizer: Marginal signal confidence ({confidence:.2f}) scaled size by {confidence_scale:.2f}")
            
        # 4. Enforce strict lot constraints (broker-defined bounds)
        final_lots = float(np.round(base_lots, 2))  # Round to 2 decimal places (standard MT5 step)
        
        if final_lots < self.min_lot:
            logger.debug(f"PositionSizer: Calculated lots {final_lots} below min. Forcing min_lot={self.min_lot}")
            final_lots = self.min_lot
            
        if final_lots > self.max_lot:
            logger.warning(f"PositionSizer: Calculated lots {final_lots} above max. Forcing max_lot={self.max_lot}")
            final_lots = self.max_lot
            
        return final_lots
