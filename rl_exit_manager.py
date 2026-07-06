import logging
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
import numpy as np
from shared_state import SharedState

logger = logging.getLogger("cheetah_rl")

class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int = 5, action_size: int = 2):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_size),
            nn.Tanh()  # Squashes continuous output to [-1.0, 1.0]
        )

    def forward(self, x):
        return self.fc(x)


class RLExitManager:
    def __init__(self, state_size: int = 5, budget_atr_min: float = 0.5, budget_atr_max: float = 3.0):
        """
        Safety-bounded Reinforcement Learning exit manager.
        The RL policy outputs actions in [-1.0, 1.0], which are strictly constrained
        by the external hard-coded safety wrapper.
        """
        self.state_size = state_size
        self.budget_min = budget_atr_min
        self.budget_max = budget_atr_max
        
        self.policy = PolicyNetwork(state_size=state_size, action_size=2)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

    def evaluate_policy(self, current_pnl: float, elapsed_bars: float, atr: float, 
                         regime_idx: float, win_rate: float) -> np.ndarray:
        """Runs inference through the PyTorch policy network."""
        state_tensor = torch.tensor([current_pnl, elapsed_bars, atr, regime_idx, win_rate], dtype=torch.float32)
        self.policy.eval()
        with torch.no_grad():
            raw_action = self.policy(state_tensor.unsqueeze(0)).squeeze(0).numpy()
        return raw_action

    def get_safe_action(self, current_pnl: float, elapsed_bars: float, atr: float, 
                        regime_idx: float, win_rate: float, position: dict, 
                        shared_state: SharedState) -> dict:
        """
        Wraps raw policy outputs in hard safety constraints.
        Overrides policy completely if risk limits are triggered.
        """
        # 1. Hard Kill Switch Override Checks
        is_halted = shared_state.get("is_halted", False)
        if is_halted:
            logger.warning("RLExitManager: Risk Halt active. RL policy overridden to Hold.")
            return {
                "action": "hold",
                "reason": "Risk manager halt override",
                "sl_adjustment": 0.0,
                "partial_close_fraction": 0.0
            }

        # 2. Query Policy Network
        raw_action = self.evaluate_policy(current_pnl, elapsed_bars, atr, regime_idx, win_rate)
        
        # raw_action[0]: Trailing stop modification (-1.0 to 1.0)
        # raw_action[1]: Partial close request (-1.0 to 1.0)
        
        # 3. Trailing Stop Safety Shell
        # Map raw continuous action from [-1, 1] to stop loss range [0.5 * atr, 3.0 * atr]
        # offset = min_bound + (normalized_val * range)
        raw_val_stop = float(raw_action[0])
        normalized_val = (raw_val_stop + 1.0) / 2.0  # Mapped to [0.0, 1.0]
        sl_atr_multiple = self.budget_min + (normalized_val * (self.budget_max - self.budget_min))
        
        # Sane bounds check: clamp final ATR multiple to [0.5, 3.0] regardless of model failures
        sl_atr_multiple = float(np.clip(sl_atr_multiple, self.budget_min, self.budget_max))
        
        # 4. Partial Close Sizing Safety Shell
        raw_val_close = float(raw_action[1])
        partial_close_fraction = 0.0
        
        # Only request partial close if model confidence exceeds 0.5
        # and position is not already partially closed
        if raw_val_close > 0.5 and not position.get("partially_closed", False):
            # Cap partial close to exactly 0.5 fraction of the initial lot size
            partial_close_fraction = 0.5
            
        action_name = "modify_exit"
        if partial_close_fraction > 0.0:
            action_name = "partial_close"
            
        return {
            "action": action_name,
            "reason": "RL safety-bounded suggestion",
            "sl_adjustment": sl_atr_multiple,
            "partial_close_fraction": partial_close_fraction
        }
