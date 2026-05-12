import logging
import pandas as pd
import time
from typing import Dict, List

from .data_collector import get_ohlcv, SYMBOL_MAP
from .feature_engine.feature_engine import FeatureEngine
from .model_ensemble.aggregator import EnsembleAggregator

log = logging.getLogger(__name__)

def institutional_retrain():
    """Syncs the entire 50-model ensemble (10 pairs x 5 timeframes)."""
    from .walk_forward_trainer import WalkForwardTrainer
    from .data_collector import SYMBOL_MAP
    
    wf_trainer = WalkForwardTrainer()
    fe = FeatureEngine()
    symbols = list(SYMBOL_MAP.keys())
    timeframes = ["1mo", "1d", "1h", "15m", "5m"]
    
    for symbol in symbols:
        for tf in timeframes:
            log.info(f"🔄 Syncing {symbol} [{tf}]...")
            
            # 1. Load Data
            df = get_ohlcv(symbol, tf, limit=3000) # Increased limit for walk-forward
            
            # Auto-fetch if missing or too small
            if df.empty or len(df) < (200 if tf == "1mo" else 1000):
                log.info(f"  📥 Data missing/insufficient in DB. Fetching {tf} for {symbol}...")
                from .data_collector import _fetch_institutional, store_ohlcv
                df_new = _fetch_institutional(symbol, tf, initial=True)
                if df_new is not None:
                    store_ohlcv(symbol, tf, df_new)
                    log.info(f"  ✅ Fetched {len(df_new)} rows.")
                    df = get_ohlcv(symbol, tf, limit=3000)

            if df.empty or len(df) < 500:
                log.warning(f"⏩ Skipping {symbol} [{tf}]: Insufficient data ({len(df)} rows).")
                continue
                
            # 2. Generate Institutional Features
            df_feat = fe.generate_features(df)
            
            # 3. Create Target
            future_ret = df_feat['close'].shift(-1).pct_change().shift(-1)
            y = pd.Series(index=df_feat.index, data=1)
            y[future_ret > 0.001] = 2
            y[future_ret < -0.001] = 0
            
            # 4. Walk-Forward Cycle (Section 4)
            # We use the most recent 24m window for the final model
            X = df_feat.iloc[:-1]
            y = y.iloc[:-1]
            
            log.info(f"  Institutional Dataset: {X.shape[1]} features, {X.shape[0]} samples.")
            
            # 5. Train Ensemble Members
            agg = EnsembleAggregator(symbol, tf)
            for model in agg.models:
                log.info(f"  Training {model.name}...")
                try:
                    model.train(X, y)
                    model.save()
                except Exception as e:
                    log.error(f"  ❌ Failed to train {model.name}: {e}")
                
            log.info(f"✅ {symbol} [{tf}] v2 Synchronized.")
            time.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    institutional_retrain()
