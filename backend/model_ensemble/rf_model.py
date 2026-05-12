from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from .base_model import InstitutionalModel

class RandomForestModel(InstitutionalModel):
    """Institutional Random Forest ensemble member."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__("random_forest", symbol, timeframe)
        self.params = {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 5,
            'n_jobs': -1,
            'random_state': 42
        }

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            return np.zeros((len(X), 3))
        return self.model.predict_proba(X)
