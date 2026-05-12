"""
verify_institutional_filtering.py — Final Audit for Phase 4.
Verifies that the prediction engine correctly applies all 3 institutional layers:
1. Primary Direction (Layer 1)
2. Meta-Model Confidence (Layer 2)
3. Market Regime Context (Layer 3)
"""
import logging
from backend.prediction_engine import predict
from backend.data_collector import ALL_SYMBOLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def run_audit():
    # Test assets that we know have been trained
    test_suite = [("EURUSD", "1h"), ("EURUSD", "1d")]
    
    print("\n" + "="*60)
    print("INSTITUTIONAL 3-LAYER FILTERING AUDIT")
    print("="*60)
    
    for symbol, tf in test_suite:
        print(f"\n--- Auditing {symbol} [{tf}]:")
        try:
            result = predict(symbol, tf)
            if not result:
                print(f"  [SKIP] No model found/data insufficient for {symbol} {tf}")
                continue
            
            print(f"  [Layer 1] Primary Signal  : {result.signal} (Prob: {result.probability:.4f})")
            print(f"  [Layer 2] Meta-Model Score: {result.meta_score:.4f} ({result.meta_decision})")
            print(f"  [Layer 3] Market Regime   : {result.market_regime} (Strength: {result.regime_strength:.2f})")
            
            status = "PASSED" if result.is_actionable() else "REJECTED (High Institutional Standard)"
            print(f"  --- FINAL DECISION: {status} ---")
            
        except Exception as e:
            print(f"  [ERROR] Audit error for {symbol} {tf}: {str(e).encode('ascii', 'ignore').decode()}")

if __name__ == "__main__":
    run_audit()
