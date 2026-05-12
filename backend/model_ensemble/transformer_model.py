import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from .base_model import InstitutionalModel

log = logging.getLogger(__name__)

class TransformerPredictor(nn.Module):
    """Simplified Transformer for institutional time-series classification."""
    def __init__(self, input_dim: int, nhead: int = 5, num_layers: int = 2):
        super(TransformerPredictor, self).__init__()
        # Ensure d_model is divisible by nhead
        valid_nhead = 1
        for i in range(nhead, 0, -1):
            if input_dim % i == 0:
                valid_nhead = i
                break
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=valid_nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: [Batch, Seq=1, Features]
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return self.softmax(out)

class TransformerModel(InstitutionalModel):
    """Institutional Transformer ensemble member."""
    
    def __init__(self, symbol: str, timeframe: str, input_dim: int = 125):
        super().__init__("transformer", symbol, timeframe)
        self.input_dim = input_dim
        # Ensure d_model is divisible by nhead
        self.model_net = TransformerPredictor(input_dim)
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model_net.train()
        optimizer = torch.optim.Adam(self.model_net.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.FloatTensor(X.values).unsqueeze(1)
        y_tensor = torch.LongTensor(y.values)
        
        for epoch in range(10):
            optimizer.zero_grad()
            output = self.model_net(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            
        self.is_trained = True
        self.model = self.model_net

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            return np.zeros((len(X), 3))
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).unsqueeze(1)
            output = self.model(X_tensor)
            return output.numpy()
