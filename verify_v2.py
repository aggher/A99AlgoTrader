import logging
import pandas as pd
from backend.institutional_orchestrator import InstitutionalOrchestrator

logging.basicConfig(level=logging.INFO)

def test_v2_pipeline():
    print("Testing Armored Core v2 Institutional Pipeline...")
    symbol_map = {"EURUSD": "EURUSD=X"}
    orchestrator = InstitutionalOrchestrator(symbol_map)
    
    # Run a cycle
    res = orchestrator.process_symbol("EURUSD", "1h")
    print(f"\nPipeline Result: {res['status']}")
    if "signal" in res:
        print(f"Signal: {res['signal']} | AlphaScore: {res.get('alpha_score', 'N/A')}")
        print(f"Adaptive Threshold: {res.get('threshold', 'N/A')}")
        print(f"Size Multiplier: {res.get('size_multiplier', 'N/A')}")
        print(f"Macro Risk: {res.get('metadata', {}).get('macro_risk', 'N/A')}")

if __name__ == "__main__":
    test_v2_pipeline()
