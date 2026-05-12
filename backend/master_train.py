import logging
import time
from backend.train_xgboost_model import train as train_l1, DEFAULT_TIMEFRAMES, MODELS_DIR
from backend.train_meta_model import train_meta as train_l2
from backend.data_collector import ALL_SYMBOLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def master_train():
    # Process symbols in order. EURUSD is mostly done, but we'll re-verify.
    t0 = time.time()
    log.info(f"🚀 Starting Master Institutional training for: {ALL_SYMBOLS}")
    
    # Preferred timeframes for institutional signals
    target_timeframes = ["1mo", "1d", "1h", "15m", "5m"]
    
    for symbol in ALL_SYMBOLS:
        for tf in target_timeframes:
            log.info(f"--- 🧱 Retraining {symbol} [{tf}] ---")
            
            # Layer 1: Primary Model (Forced)
            try:
                train_l1(symbols=[symbol], timeframes=[tf], tune=False)
            except Exception as e:
                log.error(f"Error training L1 for {symbol} {tf}: {e}")
                continue
                
            # Layer 2: Meta-Model (Forced)
            try:
                train_l2(symbol, tf)
            except Exception as e:
                log.error(f"Error training L2 for {symbol} {tf}: {e}")
                continue
                
    total_time = (time.time() - t0) / 60
    log.info(f"✅ Master training complete in {total_time:.1f} minutes.")

if __name__ == "__main__":
    master_train()
