import logging
from backend.prediction_engine import predict

logging.basicConfig(level=logging.INFO)

def verify():
    symbol, tf = "EURUSD", "1h"
    print(f"Testing 3-Layer Prediction for {symbol} {tf}...")
    
    res = predict(symbol, tf)
    if not res:
        print("Prediction failed (check model existence).")
        return
        
    print("\n--- 3-Layer Prediction Result ---")
    print(f"Signal: {res.signal} (Prob: {res.probability:.4f})")
    print(f"Layer 2 Meta-Score: {res.meta_score:.4f} ({res.meta_decision})")
    print(f"Layer 3 Regime: {res.market_regime} (Strength: {res.regime_strength})")
    print(f"Is Actionable: {res.is_actionable()}")
    
    if res.is_actionable():
        print("Actionable: Signal passed all 3 filters!")
    else:
        print("Not Actionable: Filter rejected signal (expected behavior for high-standard filtering).")

if __name__ == "__main__":
    verify()
