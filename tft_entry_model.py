import logging
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

logger = logging.getLogger("cheetah_lstm")

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, num_classes: int = 3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last sequence step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMEntryModel:
    def __init__(self, seq_len: int = 10, hidden_size: int = 32, num_layers: int = 2, 
                 epochs: int = 5, batch_size: int = 32, lr: float = 0.005):
        """
        Sequence-based entry model wrapper using PyTorch LSTM.
        Slices tabular features into time-series windows of W bars.
        """
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.model = None
        self.feature_cols = None

    def _create_sequences(self, X: pd.DataFrame, y: np.ndarray = None):
        """Slices the feature matrices and targets into sequence windows."""
        X_arr = X.values
        X_seq = []
        y_seq = []
        
        for i in range(self.seq_len - 1, len(X)):
            X_seq.append(X_arr[i - self.seq_len + 1 : i + 1])
            if y is not None:
                y_seq.append(y[i])
                
        X_seq = np.array(X_seq, dtype=np.float32)
        if y is not None:
            return X_seq, np.array(y_seq, dtype=np.int64)
        return X_seq

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Trains the LSTM classifier on sequence windows."""
        self.feature_cols = list(X.columns)
        input_size = len(self.feature_cols)
        
        # 1. Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        if len(X_seq) == 0:
            logger.error("LSTMEntryModel: Insufficient rows to build sequence windows.")
            return self
            
        # 2. Build model
        self.model = LSTMClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=3
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 3. Training Loop
        self.model.train()
        num_samples = len(X_seq)
        
        logger.info(f"LSTMEntryModel: Commencing fit loop on {num_samples} sequences...")
        for epoch in range(self.epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            epoch_loss = 0.0
            for start_idx in range(0, num_samples, self.batch_size):
                batch_indices = indices[start_idx : start_idx + self.batch_size]
                
                batch_X = torch.tensor(X_seq[batch_indices])
                batch_y = torch.tensor(y_seq[batch_indices])
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
                
            logger.debug(f"LSTM Train Epoch {epoch+1}/{self.epochs} | Loss: {epoch_loss/num_samples:.4f}")
            
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns class probabilities (shape: len(X), 3). Pads initial steps with neutral."""
        if self.model is None or self.feature_cols is None:
            raise ValueError("LSTMEntryModel: Model is not fitted yet.")
            
        self.model.eval()
        probabilities = np.zeros((len(X), 3))
        # Initial W-1 steps get default neutral predictions (class 0: 1.0 probability)
        probabilities[: self.seq_len - 1, 0] = 1.0
        
        # Create sequences for prediction
        X_seq = self._create_sequences(X)
        if len(X_seq) == 0:
            return probabilities
            
        with torch.no_grad():
            tensor_X = torch.tensor(X_seq)
            outputs = self.model(tensor_X)
            # Apply softmax
            probs = torch.softmax(outputs, dim=1).numpy()
            
        probabilities[self.seq_len - 1 :] = probs
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns class predictions (shape: len(X)). Pads initial steps with class 0."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
