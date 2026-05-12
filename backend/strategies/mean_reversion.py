import logging
import pandas as pd
from .base_strategy import BaseStrategy
from typing import Dict, Any

log = logging.getLogger(__name__)

class MeanReversionStrategy(BaseStrategy):
    """
    Section 5: Mean Reversion Strategy Module.
    Uses Bollinger Band deviations and extreme RSI.
    """
    def __init__(self):
        super().__init__("MeanReversion")

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates signals based on Price vs Bollinger Bands and RSI levels.
        """
        last_close = df['close'].iloc[-1]
        bb_hi_20 = df['vol_bb_hi_20'].iloc[-1] # This is (Price - BB_hi) / Price
        bb_lo_20 = df['vol_bb_lo_20'].iloc[-1] # This is (Price - BB_lo) / Price
        last_rsi = df['mom_rsi_14'].iloc[-1]
        
        signal = "HOLD"
        confidence = 0.5
        
        # Mean Reversion: Buy at oversold (RSI < 30) AND at or below BB Lower
        if last_rsi < 30 and bb_lo_20 <= 0:
            signal = "BUY"
            confidence = 0.80
        # Mean Reversion: Sell at overbought (RSI > 70) AND at or above BB Upper
        elif last_rsi > 70 and bb_hi_20 >= 0:
            signal = "SELL"
            confidence = 0.80
            
        return {
            "strategy": self.name,
            "signal": signal,
            "confidence": confidence,
            "metadata": {"bb_hi": bb_hi_20, "bb_lo": bb_lo_20, "rsi": last_rsi}
        }
