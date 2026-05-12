import logging
from typing import List, Dict, Any

log = logging.getLogger(__name__)

class PortfolioAllocator:
    """
    Section 5: Strategy Porterfolio Allocator.
    Determines which signals from independent strategies are executed.
    Ranks and cleans signals based on consensus and confidence.
    """

    def __init__(self):
        pass

    def allocate(self, strategy_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Coordinates multiple strategy signals into a unified portfolio decision.
        
        :param strategy_outputs: List of signal dicts from different strategies.
        """
        if not strategy_outputs:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "NO_STRATEGY_INPUT"}

        # Filtering active signals
        active_signals = [s for s in strategy_outputs if s["signal"] != "HOLD"]
        
        if not active_signals:
            return {"signal": "HOLD", "confidence": 0.5, "reason": "ALL_STRATEGIES_HOLD"}

        # Consensus Voting
        buys = [s for s in active_signals if s["signal"] == "BUY"]
        sells = [s for s in active_signals if s["signal"] == "SELL"]
        
        # Determine winning direction
        if len(buys) > len(sells):
            winner = "BUY"
            conf = sum([s["confidence"] for s in buys]) / len(strategy_outputs)
        elif len(sells) > len(buys):
            winner = "SELL"
            conf = sum([s["confidence"] for s in sells]) / len(strategy_outputs)
        else:
            # Conflict / Tie
            log.info("Portfolio conflict detected (Buy vs Sell tie). Holding.")
            return {"signal": "HOLD", "confidence": 0.5, "reason": "STRATEGY_CONFLICT"}

        # Boosting confidence if multiple strategies agree
        agreement_boost = len(active_signals) / len(strategy_outputs)
        final_conf = min(0.99, conf * (1.0 + agreement_boost * 0.2))

        return {
            "signal": winner,
            "confidence": final_conf,
            "strategy_count": len(active_signals),
            "strategies": [s["strategy"] for s in active_signals]
        }
