import logging
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

class AlphaScoringEngine:
    """
    Section 3: Institutional Alpha Scoring Engine.
    Combines ensemble probability, model agreement, meta-scores, and market context.
    Rank trade opportunities before signal issuance.
    """

    def __init__(self, base_threshold: float = 0.85):
        self.base_threshold = base_threshold
        # Weights as defined in Section 3
        self.weights = {
            "ensemble_prob": 0.30,
            "model_agreement": 0.20,
            "meta_model_score": 0.15,
            "regime_strength": 0.15,
            "liquidity_quality": 0.10,
            "volatility_stability": 0.10
        }

    def get_adaptive_threshold(self, market_vol: float) -> float:
        """
        Section 12: Adaptive Signal Thresholds.
        Raises the bar during extreme volatility to protect capital.
        """
        # market_vol is expected as a pct (e.g. 0.02)
        # Normal vol ~0.015-0.02. If > 0.03, we ramp up requirements.
        if market_vol > 0.03:
            return min(0.95, self.base_threshold + 0.05)
        elif market_vol < 0.01:
            return max(0.80, self.base_threshold - 0.05)
        return self.base_threshold

    def calculate_alphascore(self, inputs: Dict[str, float]) -> float:
        """
        Calculates the weighted AlphaScore based on Section 3 requirements.
        
        :param inputs: Dictionary with keys matching self.weights keys [0.0 - 1.0]
        """
        score = 0.0
        for key, weight in self.weights.items():
            val = inputs.get(key, 0.0)
            score += val * weight
            
        return score

    def validate_signal(self, inputs: Dict[str, float], threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Validates a signal through the Alpha Scoring gate.
        """
        current_threshold = threshold if threshold is not None else self.base_threshold
        
        alpha_score = self.calculate_alphascore(inputs)
        is_valid = alpha_score >= current_threshold
        
        if not is_valid:
            log.info(f"Alpha Scoring Rejected: Score {alpha_score:.4f} < {current_threshold:.4f}")
        else:
            log.info(f"Alpha Scoring Approved: Score {alpha_score:.4f} (Institutional Grade)")

        return {
            "alpha_score": alpha_score,
            "is_valid_alpha": is_valid,
            "threshold": current_threshold,
            "details": inputs
        }
