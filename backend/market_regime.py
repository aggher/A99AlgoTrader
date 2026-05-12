"""
market_regime.py — Detects if the market is trending or ranging.
"""
from __future__ import annotations
import pandas as pd
import ta
from ta.trend import ADXIndicator

def detect_market_regime(df: pd.DataFrame, adx_threshold: float = 25.0) -> pd.Series:
    """
    Classifies market as TRENDING (1) or RANGING (0).
    Uses ADX > 25 as a standard measure of trend strength.
    """
    if df.empty or len(df) < 30:
        return pd.Series(0, index=df.index)
        
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    regime = (adx > adx_threshold).astype(int)
    return regime.fillna(0)
