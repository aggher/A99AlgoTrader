import logging
import sys
import os

# Ensure the root is in the path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
from backend.prediction_engine import predict
from backend.database import init_db

def run_debug():
    init_db()
    print("--- Starting Debug Run for EURUSD 1h ---")
    r = predict("EURUSD", "1h")
    if r:
        print(f"Result: {r.signal}")
        print(f"Probability: {r.probability}")
        print(f"Probabilities (Buy/Hold/Sell-ish): {r.prob_buy}, {r.prob_hold}, {r.prob_sell}")
        print(f"Agreement: {r.agreement}")
        print(f"Actionable: {r.is_actionable()}")
        print(f"Meta Decision: {r.meta_decision}")
        print(f"Meta Score: {r.meta_score}")
        print(f"Market Regime: {r.market_regime}")
    else:
        print("Result: None (Skipped by Orchestrator)")

if __name__ == "__main__":
    run_debug()
