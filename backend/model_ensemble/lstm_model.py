import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from .base_model import InstitutionalModel

log = logging.getLogger(__name__)

class LSTMPredictor(nn.Module):
    """Simple LSTM for institutional time-series classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3) # 3 classes: Sell, Hold, Buy
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.softmax(out)

class LSTMModel(InstitutionalModel):
    """Institutional LSTM ensemble member."""
    
    def __init__(self, symbol: str, timeframe: str, input_dim: int = 125):
        super().__init__("lstm", symbol, timeframe)
        self.input_dim = input_dim
        self.model_net = LSTMPredictor(input_dim)
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        # Simplified institutional training loop
        self.model_net.train()
        optimizer = torch.optim.Adam(self.model_net.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values).unsqueeze(1) # [Batch, Seq=1, Features]
        y_tensor = torch.LongTensor(y.values)
        
        for epoch in range(10): # Quick initial sync
            optimizer.zero_grad()
            output = self.model_net(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            
        self.is_trained = True
        self.model = self.model_net # Store for saving

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            return np.zeros((len(X), 3))
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).unsqueeze(1)
            output = self.model(X_tensor)
            return output.numpy()
