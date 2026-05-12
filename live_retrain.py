import logging
import sys
import os

# Ensure backend is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.data_collector import collect_all
from backend.institutional_train import institutional_retrain

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    log = logging.getLogger("LiveRetrain")
    
    log.info("--- Starting Live Market Retraining Cycle ---")
    
    # Step 1: Force collect latest data from live sources (yfinance/twelvedata)
    log.info("Step 1/2: Collecting fresh live market data...")
    try:
        collect_all(initial=True) # Force full history sync to include latest candles
        log.info("Data collection complete.")
    except Exception as e:
        log.error(f"Data collection failed: {e}")
    
    # Step 2: Trigger the institutional ensemble training
    log.info("Step 2/2: Training institutional ensemble models...")
    try:
        institutional_retrain()
        log.info("Retraining cycle complete.")
    except Exception as e:
        log.error(f"Retraining failed: {e}")

    log.info("--- System Synchronized with Live Market ---")

if __name__ == "__main__":
    main()
