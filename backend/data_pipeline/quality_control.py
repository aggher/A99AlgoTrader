import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple

log = logging.getLogger(__name__)

class DataQualityControl:
    """Institutional-grade Data Quality Control (DQC) module."""

    @staticmethod
    def validate_snapshot(df: pd.DataFrame, ticker: str, is_delta: bool = False) -> Tuple[pd.DataFrame, dict]:
        """
        Validates incoming market data (Step 1).
        """
        stats = {
            "ticker": ticker,
            "original_rows": len(df),
            "duplicates_removed": 0,
            "outliers_detected": 0,
            "missing_aligned": 0,
            "is_valid": True,
            "error_reason": None
        }

        if df.empty or (not is_delta and len(df) < 50):
            stats["is_valid"] = False
            stats["error_reason"] = "INSUFFICIENT_DATA"
            return df, stats

        # Force lowercase for Step 1 consistency
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # 1. Duplicate Detection
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='last')]
        stats["duplicates_removed"] = initial_len - len(df)

        # 2. OHLC Consistency Check
        # High must be >= Open, Low, Close. Low must be <= all others.
        invalid_ohlc = (
            (df['high'] < df['low']) | 
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) |
            (df['low'] > df['open']) | 
            (df['low'] > df['close'])
        )
        if invalid_ohlc.any():
            stats["is_valid"] = False
            stats["error_reason"] = "OHLC_INCONSISTENCY"
            return df, stats

        # 3. Outlier Filtering (Statistical Spike Detection)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                mean = df[col].rolling(window=20).mean()
                std = df[col].rolling(window=20).std()
                z_score = (df[col] - mean) / (std + 1e-9)
                
                outliers = np.abs(z_score) > 5.0
                stats["outliers_detected"] += outliers.sum()
                df.loc[outliers, col] = np.nan
        
        # Handle Non-finite & Interpolate
        df = df.ffill().bfill()
        if df.isnull().values.any():
            stats["is_valid"] = False
            stats["error_reason"] = "NULL_VALUES_POST_CLEAN"
            return df, stats

        # 4. Gap Detection (Missing Candles)
        if len(df) > 1:
            try:
                expected_freq = pd.infer_freq(df.index)
                if expected_freq:
                    full_range = pd.date_range(start=df.index[0], end=df.index[-1], freq=expected_freq)
                    if len(full_range) > len(df) * 1.5: # Too many missing
                        stats["is_valid"] = False
                        stats["error_reason"] = "EXCESSIVE_GAPS"
                        return df, stats
                    if len(full_range) > len(df):
                        stats["missing_aligned"] = len(full_range) - len(df)
                        df = df.reindex(full_range).ffill()
            except Exception:
                pass

        return df, stats

    @staticmethod
    def check_stability(df: pd.DataFrame, is_delta: bool = False) -> bool:
        """Checks if data is stable enough for feature engineering (Step 1)."""
        # FeatureEngine needs at least 200 bars for EMA 200 and 100 for Z-scores.
        # We require 250 bars as a safe institutional buffer unless updating live.
        if df.empty or (not is_delta and len(df) < 250):
            log.warning(f"Data stability check failed: Insufficient bars ({len(df)}/250).")
            return False
        
        # Check for stale data (zero volatility)
        close_col = 'close' if 'close' in df.columns else 'Close'
        if df[close_col].std() < 1e-9:
            log.warning("Data stability check failed: Zero volatility detected.")
            return False
            
        return True
