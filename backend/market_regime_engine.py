"""
market_regime_engine.py — Institutional Market Regime Detection.
Classifies conditions as TRENDING or RANGING and evaluates liquidity.
"""
import pandas as pd
import numpy as np
import logging
from ta.trend import ADXIndicator

log = logging.getLogger(__name__)

class MarketRegimeEngine:
    @staticmethod
    def detect_regime(df: pd.DataFrame) -> dict:
        """
        Classifies the current market regime based on ADX and EMA dispersion.
        """
        if len(df) < 50:
            return {"regime": "UNKNOWN", "adx": 0, "strength": 0}

        # 1. ADX for Trend Strength
        adx_ind = ADXIndicator(df["high"], df["low"], df["close"], window=14)
        adx     = adx_ind.adx().iloc[-1]
        
        # 2. EMA Dispersion (Price vs EMA200)
        ema200 = df["close"].ewm(span=200).mean().iloc[-1]
        dist   = abs(df["close"].iloc[-1] - ema200) / ema200
        
        # 3. Decision Logic
        if adx > 25:
            regime = "TRENDING"
        elif adx < 20:
            regime = "RANGING"
        else:
            regime = "NEUTRAL"
            
        # Strength score 0.0 to 1.0
        strength = min(1.0, adx / 50.0)
        
        return {
            "regime": regime,
            "adx": round(adx, 2),
            "strength": round(strength, 2),
            "dist_ema200": round(dist, 4)
        }

    @staticmethod
    def is_favorable(regime_data: dict, signal_type: str) -> bool:
        """
        Determines if the regime favors the signal type.
        E.g., don't take trend-following signals in tight ranges.
        """
        regime = regime_data["regime"]
        if regime == "RANGING" and regime_data["strength"] < 0.15:
            # High risk of whipsaws in very low volatility ranges
            return False
            
        return True
