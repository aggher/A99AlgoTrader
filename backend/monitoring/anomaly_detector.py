import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

class AnomalyDetector:
    """Institutional Market Anomaly & Flash Crash Detection Engine."""
    
    def __init__(self):
        self.flash_crash_threshold = -0.05 # 5% drop in one bar
        self.spread_threshold_ratio = 5.0  # Spread > 5x normal

    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects catastrophic market events:
        1. Flash Crashes (Velocity-based)
        2. Liquidity Gaps (Volume/Spread)
        3. Statistical Spikes (Z-score > 10)
        """
        if df.empty or len(df) < 5:
            return {"anomaly": False, "type": "NONE"}

        # 1. Flash Crash
        returns = df['close'].pct_change()
        last_return = returns.iloc[-1]
        if last_return < self.flash_crash_threshold:
            log.critical(f"🚨 FLASH CRASH DETECTED: {last_return*100:.2f}% drop detected.")
            return {"anomaly": True, "type": "FLASH_CRASH", "impact": last_return}

        # 2. Extreme Volatility Spike
        vol_z = (df['vol_atr_20_pct'].iloc[-1] - df['vol_atr_20_pct'].rolling(100).mean().iloc[-1]) / (df['vol_atr_20_pct'].rolling(100).std().iloc[-1] + 1e-9)
        if vol_z > 8.0:
            log.error(f"⚠️ EXTREME VOLATILITY DETECTED: Z-Score {vol_z:.2f}")
            return {"anomaly": True, "type": "VOLATILITY_SPIKE", "impact": vol_z}

        # 3. Liquidity Vacuum
        rel_vol = df['liq_rel_vol_20'].iloc[-1]
        if rel_vol < 0.05:
            log.warning(f"🕳️ LIQUIDITY VACUUM DETECTED: Relative Volume {rel_vol:.4f}")
            return {"anomaly": True, "type": "LIQUIDITY_GAP", "impact": rel_vol}

        return {"anomaly": False, "type": "NONE"}
