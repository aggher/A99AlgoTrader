import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any

log = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Section 9: Institutional Model Performance Monitor.
    Tracks live metrics and dynamically adjusts ensemble influence.
    """

    def __init__(self):
        self.trade_history = []
        self.thresholds = {
            "min_sharpe": 1.2,
            "min_win_rate": 0.45,
            "max_drawdown": 0.15
        }

    def record_trade(self, trade_data: Dict[str, Any]):
        """
        Records a completed trade for performance analysis.
        """
        self.trade_history.append(trade_data)
        
    def calculate_rolling_metrics(self, window: int = 50) -> Dict[str, float]:
        """
        Calculates rolling institutional metrics (Sharpe, Win Rate, PF).
        """
        if len(self.trade_history) < 10:
            return {"sharpe": 2.0, "win_rate": 0.5, "status": "WARMING_UP"}

        recent = self.trade_history[-window:]
        pnl = [t.get("pnl", 0) for t in recent]
        
        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p <= 0]
        
        win_rate = len(wins) / len(pnl)
        
        avg_ret = np.mean(pnl)
        std_ret = np.std(pnl) + 1e-9
        sharpe = (avg_ret / std_ret) * np.sqrt(252) # Annualized proxy
        
        profit_factor = sum(wins) / (abs(sum(losses)) + 1e-9)
        
        status = "HEALTHY"
        if sharpe < self.thresholds["min_sharpe"] or win_rate < self.thresholds["min_win_rate"]:
            status = "DEGRADED"
            log.warning(f"⚠️ Performance Warning: Sharpe {sharpe:.2f}, Win Rate {win_rate:.2%}")

        return {
            "sharpe": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "status": status
        }

    def get_influence_multiplier(self) -> float:
        """
        Returns a multiplier [0.0 - 1.0] to reduce model influence if degraded.
        """
        metrics = self.calculate_rolling_metrics()
        if metrics["status"] == "DEGRADED":
            return 0.5 # Halve exposure/influence
        return 1.0
