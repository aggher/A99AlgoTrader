import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
from .base_model import InstitutionalModel

log = logging.getLogger(__name__)

class LightGBMModel(InstitutionalModel):
    """Institutional LightGBM ensemble member."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__("lightgbm", symbol, timeframe)
        self.params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multiclass',
            'num_class': 3,
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': -1
        }

    def train(self, X: pd.DataFrame, y: pd.Series):
        try:
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(X, y)
            self.is_trained = True
        except Exception as e:
            log.error(f"LightGBM Training Error: {e}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            return np.zeros((len(X), 3))
        return self.model.predict_proba(X)
