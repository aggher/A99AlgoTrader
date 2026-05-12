import logging
from backend.institutional_orchestrator import InstitutionalOrchestrator
from backend.data_collector import SYMBOL_MAP
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)

def scan_probs():
    orch = InstitutionalOrchestrator(SYMBOL_MAP)
    # Target some high-liquidity pairs
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    timeframes = ["15m", "1h", "1d"]
    
    print(f"{'Asset':<10} | {'TF':<5} | {'SELL':<8} | {'HOLD':<8} | {'BUY':<8} | {'STATUS'}")
    print("-" * 60)
    
    for sym in pairs:
        for tf in timeframes:
            try:
                df = orch.ingestor.fetch_institutional_data(sym, tf, "60d")
                if df is None: continue
                df_feat = orch.feature_engine.generate_features(df)
                
                from backend.model_ensemble.aggregator import EnsembleAggregator
                agg = EnsembleAggregator(sym, tf)
                all_probs = []
                for model in agg.models:
                    if model.load():
                        p = model.predict_proba(df_feat)
                        all_probs.append(p)
                
                if not all_probs: continue
                
                # Manual weighted average (simplified)
                weighted_avg = np.zeros(all_probs[0].shape)
                for i, p in enumerate(all_probs):
                    weighted_avg += p * 0.2 # Equal weight for scan
                
                last_p = weighted_avg[-1]
                res = orch.process_symbol(sym, tf)
                status = res.get("status", "N/A")
                
                print(f"{sym:<10} | {tf:<5} | {last_p[0]:.4f} | {last_p[1]:.4f} | {last_p[2]:.4f} | {status}")
            except Exception as e:
                pass

if __name__ == "__main__":
    scan_probs()
