import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List

log = logging.getLogger(__name__)

class DriftDetector:
    """Institutional Model Drift & Stability Monitoring Engine."""
    
    def __init__(self):
        self.drift_threshold = 0.4  # PSI > 0.4 indicates significant drift
        self.feature_baselines = {}

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculates the Population Stability Index (PSI).
        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        """
        def scale_range(data, min_val, max_val):
            return np.histogram(data, bins=buckets, range=(min_val, max_val))[0] / len(data)

        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        
        expected_percents = scale_range(expected, min_val, max_val)
        actual_percents = scale_range(actual, min_val, max_val)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi_value)

    def monitor_feature_stability(self, current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> Dict[str, float]:
        """
        Monitors stability for top institutional features.
        Returns: {feature_name: psi_score}
        """
        psi_scores = {}
        # Focus on semantic stability (normalized indicators)
        # Exclude 'atr' as it's a magnitude feature prone to regime shifts
        core_features = [c for c in current_df.columns if any(p in c for p in ['rsi', 'adx', 'ema_200_dist', 'z_score'])]
        
        for feat in core_features:
            if feat in baseline_df.columns:
                psi = self.calculate_psi(baseline_df[feat].values, current_df[feat].values)
                psi_scores[feat] = psi
                
                if psi > self.drift_threshold:
                    log.warning(f"⚠️ Stability warning for {feat}: PSI={psi:.4f}")
                    
        return psi_scores

    def get_system_health(self, psi_scores: Dict[str, float]) -> str:
        """Categorizes overall system health based on drift."""
        avg_psi = np.mean(list(psi_scores.values())) if psi_scores else 0.0
        if avg_psi < 0.1: return "STABLE"
        if avg_psi < 0.2: return "WARNING"
        return "CRITICAL_DRIFT"
