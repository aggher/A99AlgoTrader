import logging
import pandas as pd
from .base_strategy import BaseStrategy
from typing import Dict, Any

log = logging.getLogger(__name__)

class TrendFollowingStrategy(BaseStrategy):
    """
    Section 5: Trend Following Strategy Module.
    Uses EMA crossovers and momentum as base logic.
    """
    def __init__(self):
        super().__init__("TrendFollowing")

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates signals based on EMA(20) vs EMA(50) and RSI direction.
        """
        last_ema20 = df['ema_20'].iloc[-1]
        last_ema50 = df['ema_50'].iloc[-1]
        last_rsi = df['mom_rsi_14'].iloc[-1]
        
        signal = "HOLD"
        confidence = 0.5
        
        if last_ema20 > last_ema50 and last_rsi > 50:
            signal = "BUY"
            confidence = 0.75
        elif last_ema20 < last_ema50 and last_rsi < 50:
            signal = "SELL"
            confidence = 0.75
            
        return {
            "strategy": self.name,
            "signal": signal,
            "confidence": confidence,
            "metadata": {"ema_diff": last_ema20 - last_ema50, "rsi": last_rsi}
        }
