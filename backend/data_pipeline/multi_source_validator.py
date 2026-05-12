import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

log = logging.getLogger(__name__)

class MultiSourceValidator:
    """
    Institutional Cross-Source Data Validation Engine.
    Verifies price, volume, and timestamp consistency across multiple providers.
    """

    def __init__(self, price_threshold: float = 0.005, vol_threshold: float = 0.15):
        """
        :param price_threshold: Max allowed % discrepancy between sources (default 0.5%)
        :param vol_threshold: Max allowed % discrepancy in volume (default 15%)
        """
        self.price_threshold = price_threshold
        self.vol_threshold = vol_threshold

    def verify_consistency(self, primary_df: pd.DataFrame, secondary_df: Optional[pd.DataFrame], symbol: str) -> Tuple[bool, str]:
        """
        Compares primary and secondary data for discrepancies.
        """
        if secondary_df is None:
            log.warning(f"Secondary source missing for {symbol}. Running on single source (reduced integrity).")
            return True, "SINGLE_SOURCE_PROCEEDED"

        # 1. Timestamp Consistency & Alignment
        if not primary_df.index.equals(secondary_df.index):
            log.info(f"Attempting robust field alignment for {symbol}...")
            # Reindex secondary to match primary with a small tolerance (30s)
            # This handles minor offset differences (e.g. 14:00:00 vs 14:00:01)
            secondary_df = secondary_df.reindex(primary_df.index, method='nearest', tolerance=pd.Timedelta('30s'))
            
            # Check for fill quality after alignment
            valid_rows = secondary_df.notna().all(axis=1).sum()
            if valid_rows < len(primary_df) * 0.85:
                return False, f"TIMESTAMP_MISMATCH: Only {valid_rows}/{len(primary_df)} rows aligned within 30s tolerance."
            
            # Fill missing aligned values if any (rare with nearest)
            secondary_df = secondary_df.ffill().bfill()

        # 2. Price Verification (Close Price)
        price_diff = np.abs(primary_df['close'] - secondary_df['close']) / (primary_df['close'] + 1e-9)
        max_price_diff = price_diff.max()
        
        if max_price_diff > self.price_threshold:
            log.error(f"Price discrepancy too high for {symbol}: {max_price_diff:.4%}")
            return False, f"PRICE_DISCREPANCY: {max_price_diff:.4%}"

        # 3. Volume Verification
        vol_diff = np.abs(primary_df['volume'] - secondary_df['volume']) / (primary_df['volume'] + 1e-9)
        max_vol_diff = vol_diff.max()

        if max_vol_diff > self.vol_threshold:
            log.warning(f"Volume discrepancy detected for {symbol}: {max_vol_diff:.4%}")
            # Volume discrepancies are common in FX, we log but don't always abort unless extreme (>50%)
            if max_vol_diff > 0.50:
                return False, f"CRITICAL_VOLUME_DISCREPANCY: {max_vol_diff:.4%}"

        log.info(f"Cross-source verification passed for {symbol}. (Price Diff: {max_price_diff:.4%})")
        return True, "VERIFIED_CONSISTENT"

    def validate_ingestion(self, symbol: str, timeframe: str, primary_data: pd.DataFrame, secondary_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        High-level entry point for Section 1 requirement.
        """
        is_consistent, reason = self.verify_consistency(primary_data, secondary_data, symbol)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "VALID" if is_consistent else "INVALID",
            "integrity_score": 1.0 if is_consistent and secondary_data is not None else 0.5,
            "reason": reason
        }
