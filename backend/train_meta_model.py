"""
train_meta_model.py — Layer 2 Training Pipeline.
Trains an XGBoost model to predict signal success (WIN/LOSS).
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

from backend.data_collector import get_ohlcv
from backend.feature_engineering import compute_features, FEATURE_COLUMNS
from backend.market_regime_engine import MarketRegimeEngine

log = logging.getLogger(__name__)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def generate_meta_dataset(symbol: str, tf: str):
    """
    Simulates signals using Layer 1 and labels them for Layer 2.
    """
    df = get_ohlcv(symbol, tf)
    if df.empty: return None
    
    df = compute_features(df)
    
    # ── MTF Enrichment for Metadata Accuracy ──
    MTF_HIERARCHY = {"1h": ["1d", "1mo"], "15m": ["1h", "1d"], "5m": ["15m", "1h"]}
    if tf in MTF_HIERARCHY:
        for htf in MTF_HIERARCHY[tf]:
            h_raw = get_ohlcv(symbol, htf)
            if not h_raw.empty:
                h_feat = compute_features(h_raw)
                from backend.feature_engineering import merge_mtf_features
                df = merge_mtf_features(df, h_feat, htf)

    # Load Primary Model (Layer 1)
    model_path = MODELS_DIR / f"{symbol}_{tf}.json"
    if not model_path.exists():
        log.error("Primary model not found for %s %s", symbol, tf)
        return None
        
    l1_model = xgb.XGBClassifier()
    l1_model.load_model(str(model_path))
    
    # Predict probabilities - Ensure feature alignment
    expected = l1_model.get_booster().feature_names
    for col in expected:
        if col not in df.columns: df[col] = 0.0
    
    X = df[expected]
    probs = l1_model.predict_proba(X)
    
    # Target labeling: Success if profit > 0.5% in 20 bars
    # Using a simpler proxy for WIN/LOSS for meta-training
    future_return = df["close"].shift(-20) / df["close"] - 1
    
    meta_records = []
    for i in range(len(df) - 20):
        p_buy, p_sell = probs[i][0], probs[i][2]
        
        signal = None
        if p_buy > 0.70: signal = "BUY"
        elif p_sell > 0.70: signal = "SELL"
        
        if signal:
            # Outcome labeling
            ret = future_return.iloc[i]
            outcome = 1 if (signal == "BUY" and ret > 0.005) or (signal == "SELL" and ret < -0.005) else 0
            
            # Feature extraction (Meta-Features)
            regime = MarketRegimeEngine.detect_regime(df.iloc[:i+1])
            
            meta_feat = {
                "prob_buy": p_buy,
                "prob_sell": p_sell,
                "rsi": df["rsi"].iloc[i],
                "vol_brk": df["volatility_breakout"].iloc[i],
                "vol_imb": df["volume_imbalance"].iloc[i],
                "regime_strength": regime["strength"],
                "target": outcome
            }
            meta_records.append(meta_feat)
            
    return pd.DataFrame(meta_records)

def train_meta(symbol: str, tf: str):
    log.info("Generating Meta-Dataset for %s %s...", symbol, tf)
    data = generate_meta_dataset(symbol, tf)
    if data is None or len(data) < 20:
        log.warning("Not enough signals to train meta-model.")
        return

    log.info("Training Meta-Model (Layer 2) with %d signals...", len(data))
    X = data.drop("target", axis=1)
    y = data["target"]
    
    # Simple XGBoost for Meta-Layer
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    save_path = MODELS_DIR / f"{symbol}_{tf}_meta.json"
    model.save_model(str(save_path))
    log.info("Meta-Model saved to %s", save_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_meta("EURUSD", "1h")
