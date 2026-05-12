import logging
from typing import Dict, List, Any
import pandas as pd

log = logging.getLogger(__name__)

class RiskManager:
    """Institutional Portfolio Risk Management Engine."""
    
    def __init__(self, initial_equity: float = 100000.0):
        self.equity = initial_equity
        self.max_trade_risk_pct = 0.01  # 1% per trade
        self.max_daily_drawdown = 0.02  # 2% daily limit
        self.max_total_drawdown = 0.05  # 5% hard stop (Safe Mode)
        
        # Exposure Limits
        self.max_correlated_exposure = 0.25 # Max 25% in one currency group
        
        self.current_drawdown = 0.0
        self.is_safe_mode = False

    def calculate_dynamic_size(self, confidence: float, atr_pct: float) -> float:
        """
        Section 7: Dynamic Position Sizing Engine.
        PositionSize = Capital * RiskFactor * SignalConfidence / MarketVolatility
        """
        if atr_pct <= 0:
            return 1.0 # Fallback
            
        # Standardize: higher volatility = smaller size
        # We use a base multiplier (e.g. 0.02 as typical ATR)
        vol_scalar = 0.002 / (atr_pct + 1e-9) 
        
        # Adaptive Multiplier [0.1 to 2.0]
        dynamic_mult = confidence * vol_scalar
        return max(0.1, min(2.0, dynamic_mult))

    def validate_signal(self, signal_data: Dict[str, Any], current_positions: List[Dict]) -> Dict[str, Any]:
        """
        Validates a signal against institutional risk constraints.
        Includes Section 10: Portfolio Exposure Intelligence.
        """
        if self.is_safe_mode:
            return {"approved": False, "reason": "SAFE_MODE_ACTIVE", "size": 0.0}

        # 1. Total Drawdown Check
        if self.current_drawdown >= self.max_total_drawdown:
            self.is_safe_mode = True
            return {"approved": False, "reason": "MAX_DRAWDOWN_REACHED", "size": 0.0}

        symbol = signal_data.get("symbol", "")
        confidence = signal_data.get("confidence", 0.5)
        atr_pct = signal_data.get("atr_pct", 0.002)

        # 2. Section 10: Portfolio Exposure Intelligence
        # Monitor Currency Groups (Correlation Clusters)
        base_ccy = symbol[:3]
        cluster_exposure = sum([p.get('exposure', 0.0) for p in current_positions if p['symbol'].startswith(base_ccy)])
        
        if cluster_exposure > (self.equity * self.max_correlated_exposure):
            log.warning(f"Risk Reject: Correlated concentration in {base_ccy}")
            return {"approved": False, "reason": "CORRELATION_CLUSTER_LIMIT"}

        # 3. Section 7: Dynamic Position Sizing
        size_multiplier = self.calculate_dynamic_size(confidence, atr_pct)
        risk_amount = self.equity * self.max_trade_risk_pct * size_multiplier
        
        # Upper Bound: 5% of equity per trade MAX
        risk_amount = min(risk_amount, self.equity * 0.05)

        return {
            "approved": True, 
            "reason": "OK", 
            "size_multiplier": size_multiplier,
            "equity_at_risk": risk_amount,
            "exposure": risk_amount * (1 / 0.02) # Proxy for leveraged exposure
        }

    def update_account_state(self, equity: float, drawdown: float):
        """Updates internal risk state from live account data."""
        self.equity = equity
        self.current_drawdown = drawdown
        if drawdown >= self.max_total_drawdown:
            self.is_safe_mode = True
