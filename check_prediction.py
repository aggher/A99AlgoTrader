
import logging
import sys
import os

# Add parent dir to path
sys.path.append(os.getcwd())

from backend.data_collector import collect_all
from backend.prediction_engine import run_all
from backend.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def test_cycle():
    init_db()
    print("--- Starting Manual Data Collection (2d window) ---")
    collect_all(initial=False)
    
    print("\n--- Starting Manual Prediction Cycle ---")
    signals = run_all()
    
    print(f"\n--- Cycle Done: Generated {len(signals)} signals ---")
    for s in signals:
        if s.signal != "HOLD":
            print(f"Actionable: {s.symbol} {s.timeframe} -> {s.signal} ({s.probability:.2f})")
        else:
            # Print a few holds to confirm it works
            if s.symbol == "EURUSD":
                print(f"Hold: {s.symbol} {s.timeframe} -> {s.signal} ({s.probability:.2f})")

if __name__ == "__main__":
    test_cycle()
