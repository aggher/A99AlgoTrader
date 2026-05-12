import logging
import pandas as pd
from typing import Dict, Any

log = logging.getLogger(__name__)

class LiquidityMonitor:
    """Step 8: Liquidity and Spread Protection."""
    
    def __init__(self):
        self.max_spread_pct = 0.0005 # 0.05% max allowed spread
        self.min_volume_zscore = -1.5 # Reject if volume is significantly below average
        
    def check_conditions(self, symbol: str, df_feat: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates execution conditions based on liquidity and spread.
        """
        if df_feat.empty:
            return {"safe": False, "reason": "NO_DATA"}
            
        # 1. Spread Check (Simulated based on volatility if real-time spread not available)
        # In a real system, this would use live bid-ask data.
        # Here we simulate spread as a function of volatility (z-score)
        vol_z = df_feat['vol_atr_20_pct'].iloc[-1]
        sim_spread_pct = 0.0002 + (max(0, vol_z) * 0.0001)
        
        if sim_spread_pct > self.max_spread_pct:
            return {"safe": False, "reason": "EXCESSIVE_SPREAD", "val": sim_spread_pct}
            
        # 2. Volume/Liquidity Check
        # Uses 'liq_rel_vol_20' (Relative Volume) if available
        if 'liq_rel_vol_20' in df_feat.columns:
            rel_vol = df_feat['liq_rel_vol_20'].iloc[-1]
            vol_val = df_feat['volume'].iloc[-1] if 'volume' in df_feat.columns else 0
            if rel_vol < 0.3 and vol_val > 0: # Only reject if volume data exists
                return {"safe": False, "reason": "INSUFFICIENT_LIQUIDITY", "val": rel_vol}
                
        return {"safe": True, "reason": "OK"}
