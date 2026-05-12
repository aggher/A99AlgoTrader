import logging
from backend.institutional_train import institutional_retrain
from backend.data_collector import SYMBOL_MAP

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Train only EURUSD 1h for proof of concept
    print("Running Institutional V2 Retraining (Proof of Concept: EURUSD 1h)...")
    from unittest.mock import patch
    
    # Patch symbols/timeframes to be fast
    with patch('backend.institutional_train.SYMBOL_MAP', {"EURUSD": "EURUSD=X"}):
        from backend import institutional_train
        institutional_train.timeframes = ["1h"]
        institutional_retrain()
