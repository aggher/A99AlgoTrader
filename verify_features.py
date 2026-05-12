import pandas as pd
import numpy as np
import logging
from backend.data_collector import get_ohlcv
from backend.feature_engineering import compute_features, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)

def verify():
    symbol, tf = "EURUSD", "1h"
    print(f"Fetching data for {symbol} {tf}...")
    df = get_ohlcv(symbol, tf)
    if df.empty:
        print("No data found.")
        return

    print(f"Computing {len(FEATURE_COLUMNS)} features...")
    df_feat = compute_features(df)
    
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df_feat.columns]
    if missing_cols:
        print(f"MISSING COLUMNS: {missing_cols}")
    else:
        print("All feature columns present.")

    nan_counts = df_feat[FEATURE_COLUMNS].isna().sum()
    total_nan = nan_counts.sum()
    
    if total_nan > 0:
        print("NaN detected in features:")
        print(nan_counts[nan_counts > 0])
    else:
        print("Success: 0 NaNs in feature set.")
    
    print("\nFeature Samples (last 5 rows):")
    cols_to_show = [c for c in FEATURE_COLUMNS[:5]] + [c for c in FEATURE_COLUMNS[-5:]]
    print(df_feat[cols_to_show].tail())

if __name__ == "__main__":
    verify()
