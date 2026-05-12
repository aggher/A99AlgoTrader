import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any
from .base_model import InstitutionalModel
from .xgb_model import XGBoostModel
from .rf_model import RandomForestModel
from .lgbm_model import LightGBMModel
from .lstm_model import LSTMModel
from .transformer_model import TransformerModel

log = logging.getLogger(__name__)

class EnsembleAggregator:
    """Combines predictions from multiple institutional models."""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.models: List[InstitutionalModel] = []
        
        # Initialize available models
        self.models.append(XGBoostModel(symbol, timeframe))
        self.models.append(RandomForestModel(symbol, timeframe))
        self.models.append(LightGBMModel(symbol, timeframe))
        self.models.append(LSTMModel(symbol, timeframe))
        self.models.append(TransformerModel(symbol, timeframe))
        # (LGBM, LSTM, Transformer added here as implemented)

        # Institutional Model Weights (Step 6)
        self.weights = {
            'xgboost': 0.30,
            'lightgbm': 0.25,
            'random_forest': 0.20,
            'lstm': 0.15,
            'transformer': 0.10
        }

    def aggregate_predictions(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes the weighted consensus across the ensemble.
        Returns: {final_signal, final_probability, model_agreement_score}
        """
        all_probs = []
        active_weights = []
        
        for model in self.models:
            if model.load():
                probs = model.predict_proba(X)
                all_probs.append(probs)
                active_weights.append(self.weights.get(model.name, 0.1))
        
        if not all_probs:
            return {"signal": "HOLD", "prob": 0.0, "agreement": 0.0}

        # Normalize weights
        active_weights = np.array(active_weights)
        active_weights /= active_weights.sum()
        
        # Weighted Average Probabilities
        weighted_avg = np.zeros((all_probs[0].shape))
        for i, prob in enumerate(all_probs):
            if prob.shape == weighted_avg.shape:
                weighted_avg += prob * active_weights[i]
            
        # Agreement Score (Step 5)
        # 4 out of 5 models agreeing = 0.8 agreement
        
        # Determine highest prob class for each model (last row)
        model_predictions = [np.argmax(p[-1]) for p in all_probs]
        final_idx = np.argmax(weighted_avg[-1])
        
        # Agreement Score (Step 5): Count how many models agree with the Weighted Consensus
        agree_count = sum([1 for p in model_predictions if p == final_idx])
        agreement_score = agree_count / len(all_probs) if all_probs else 0.0
        
        final_prob = np.max(weighted_avg[-1])
        signals = ['SELL', 'HOLD', 'BUY']
        final_signal = signals[final_idx]

        # Step 5 & 6 Filters
        # Only allow BUY/SELL if thresholds are met
        if final_signal != 'HOLD':
            if agreement_score < 0.70 or final_prob < 0.75:
                log.info(f"Signal {final_signal} rejected: Agreement={agreement_score:.2f}, Prob={final_prob:.4f}")
                final_signal = 'HOLD'
        else:
            log.info(f"Consensus is HOLD (Agreement={agreement_score:.2f}, Prob={final_prob:.4f})")
        
        return {
            "signal": final_signal,
            "prob": final_prob,
            "agreement": float(agreement_score),
            "raw_probs": weighted_avg[-1].tolist()
        }
