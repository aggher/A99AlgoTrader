import logging
import sys
import os

# Ensure the root is in the path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.ERROR) # Suppress warnings
from backend.prediction_engine import predict
ALL_SYMBOLS_DEBUG = ["EURUSD", "GBPUSD", "XAUUSD"]
from backend.database import init_db

def scan_all():
    init_db()
    tfs = ["1d", "1h", "15m", "5m"]
    print(f"Scanning {len(ALL_SYMBOLS_DEBUG)} symbols across {len(tfs)} timeframes...\n")
    
    found = 0
    for symbol in ALL_SYMBOLS_DEBUG:
        for tf in tfs:
            r = predict(symbol, tf)
            if r:
                print(f"[{symbol} {tf}] -> {r.signal} (Prob: {r.probability}, Agreement: {r.agreement}, Actionable: {r.is_actionable()})")
                if r.signal != "HOLD":
                    found += 1
            else:
                # Need to check why it's None. The orchestrator rejected it.
                print(f"[{symbol} {tf}] -> SKIP (Consensus was likely HOLD)")
    
    if found == 0:
        print("Done. No BUY/SELL signals found and all are HOLD.")
    else:
        print(f"Done. Found {found} non-HOLD signals.")

if __name__ == "__main__":
    scan_all()
