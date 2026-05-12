import xgboost as xgb
import pandas as pd
import numpy as np
from .base_model import InstitutionalModel

class XGBoostModel(InstitutionalModel):
    """Institutional XGBoost ensemble member."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__("xgboost", symbol, timeframe)
        self.params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'num_class': 3,
            'n_jobs': -1,
            'random_state': 42
        }

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            return np.zeros((len(X), 3))
        return self.model.predict_proba(X)
