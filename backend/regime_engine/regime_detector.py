import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

class RegimeDetector:
    """Institutional Market Regime Engine."""
    
    def __init__(self):
        self.regimes = {
            0: "LOW_VOL_SIDEWAYS",
            1: "TRENDING_BULLISH",
            2: "TRENDING_BEARISH",
            3: "HIGH_VOL_CHOPPY",
            4: "VOLATILITY_EXPANSION"
        }

    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects the current market regime based on multi-timeframe heuristics:
        1. ADX (Trend strength)
        2. EMA alignment (Trend direction)
        3. ATR Normalized Volatility
        4. Hurst Exponent (Mean Reversion vs Trending)
        """
        if df.empty or len(df) < 50:
            return {"regime": "UNKNOWN", "id": -1}

        close = df['close']
        last_adx = df['trend_adx'].iloc[-1]
        last_atr_pct = df['vol_atr_20_pct'].iloc[-1]
        last_hurst = df['trend_hurst'].iloc[-1]
        
        # 1. Volatility check
        vol_avg = df['vol_atr_20_pct'].rolling(100).mean().iloc[-1]
        is_high_vol = last_atr_pct > (vol_avg * 1.5)
        
        # 2. Trend check
        ema_f = df['ema_20'].iloc[-1]
        ema_s = df['ema_200'].iloc[-1]
        
        is_trending = last_adx > 25
        is_bullish = ema_f > ema_s
        
        # Logic Tree
        if is_high_vol:
            regime_id = 3 if last_adx < 20 else 4
        elif is_trending:
            regime_id = 1 if is_bullish else 2
        else:
            regime_id = 0
            
        return {
            "regime": self.regimes.get(regime_id, "UNKNOWN"),
            "id": regime_id,
            "adx": float(last_adx),
            "vol_relative": float(last_atr_pct / vol_avg),
            "hurst": float(last_hurst)
        }
