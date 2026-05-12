from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import os
import joblib
import logging
from typing import Dict, Any, Tuple

log = logging.getLogger(__name__)

class InstitutionalModel(ABC):
    """Base class for all institutional-grade AI models."""
    
    def __init__(self, name: str, symbol: str, timeframe: str):
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self.is_trained = False
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.models_dir = os.path.join(base_dir, "models", "ensemble", symbol, timeframe)
        os.makedirs(self.models_dir, exist_ok=True)

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def save(self):
        if self.model:
            path = os.path.join(self.models_dir, f"{self.name}.joblib")
            joblib.dump(self.model, path)
            log.info(f"Model {self.name} saved to {path}")

    def load(self) -> bool:
        path = os.path.join(self.models_dir, f"{self.name}.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True
            log.info(f"Model {self.name} loaded from {path}")
            return True
        return False
