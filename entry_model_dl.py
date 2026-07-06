import logging
import numpy as np
import pandas as pd

# CRITICAL WARNING: As per Lopez de Prado and quantitative best practices, 
# the LightGBM/CatBoost tree-based baseline model should be fully validated and proved 
# to have predictive power before investing time in deep sequence learning models. 
# Deep learning models on tabular financial data are highly prone to overfitting, 
# require massive volumes of data, and are extremely sensitive to hyperparameter choices.

logger = logging.getLogger(__name__)

# Try to import torch safely
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not found. entry_model_dl.py will fall back to placeholder behavior.")

class LSTMClassifier(nn.Module if HAS_TORCH else object):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 3):
        if not HAS_TORCH:
            return
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Gather output of the last sequence step
        last_step_out = lstm_out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits

class DeepEntryModel:
    def __init__(self, input_dim: int = 10, seq_len: int = 60, use_deep_model: bool = False):
        """
        PyTorch LSTM entry model sharing the exact same interface as EntryModel.
        Uses Deep Learning if use_deep_model=True and torch is available.
        Otherwise falls back or logs placeholder behavior.
        """
        self.use_deep_model = use_deep_model
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model = None
        self.feature_cols = []
        
        if self.use_deep_model and not HAS_TORCH:
            logger.error("Deep model enabled in config, but PyTorch is not installed. Forcing fallback.")
            self.use_deep_model = False

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, seq_len: int = 60, epochs: int = 10, batch_size: int = 32):
        """
        Fits the LSTM model on sequentialized feature blocks.
        Reshapes 2D tabular features into 3D time-series sequences: (samples, seq_len, features).
        """
        self.feature_cols = list(X_train.columns)
        self.input_dim = len(self.feature_cols)
        self.seq_len = seq_len
        
        if not self.use_deep_model:
            logger.info("Deep model is disabled. Skipping PyTorch training sequence.")
            return self
            
        logger.info("Initializing PyTorch LSTM model sequence training...")
        self.model = LSTMClassifier(input_dim=self.input_dim)
        
        # Prepare 3D sequences
        X_seq, y_seq = self._create_sequences(X_train.values, y_train, self.seq_len)
        
        if len(X_seq) == 0:
            logger.warning("Insufficient samples to build 3D sequence tensors. Skipping training.")
            return self
            
        # Convert to Torch Tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"LSTM training - Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
            
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts probability distribution. 
        Outputs identical shape (n_samples, 3) as entry_model.py.
        """
        n_samples = len(X)
        if not self.use_deep_model or self.model is None:
            # Fallback placeholder returns uniform probabilities
            logger.warning("DeepEntryModel placeholder predicting uniform probabilities.")
            probs = np.ones((n_samples, 3)) / 3.0
            return probs
            
        self.model.eval()
        X_vals = X[self.feature_cols].values
        
        # Reconstruct sequences for each prediction row (lookback window of seq_len)
        # For rows < seq_len, we zero-pad
        X_padded = np.zeros((n_samples, self.seq_len, self.input_dim))
        for i in range(n_samples):
            start_idx = max(0, i - self.seq_len + 1)
            sub_window = X_vals[start_idx : i + 1]
            pad_len = self.seq_len - len(sub_window)
            X_padded[i, pad_len:, :] = sub_window
            
        X_tensor = torch.tensor(X_padded, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            # Softmax to get probabilities
            probs = torch.softmax(logits, dim=1).numpy()
            
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels: 0 (No-trade), 1 (Long), 2 (Short)."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def _create_sequences(self, X_data: np.ndarray, y_data: np.ndarray, seq_len: int):
        """Helper to create sliding 3D sequences from 2D data arrays."""
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X_data)):
            X_seq.append(X_data[i - seq_len : i])
            y_seq.append(y_data[i])
        return np.array(X_seq), np.array(y_seq)
