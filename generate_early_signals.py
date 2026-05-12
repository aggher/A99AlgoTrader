import logging
import time
from backend.prediction_engine import predict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_early_cycle():
    # Only run for synchronized pairs
    pairs = ["EURUSD", "AUDUSD"]
    timeframes = ["1h", "15m", "5m"]
    
    log.info("🚀 Starting early signal generation for synchronized pairs...")
    
    for sym in pairs:
        for tf in timeframes:
            log.info(f"Checking {sym} [{tf}]...")
            try:
                res = predict(sym, tf)
                if res:
                    log.info(f"✅ Generated Signal: {res.signal} (Conf: {res.confidence:.4f})")
                else:
                    log.info(f"⏩ No signal for {sym} [{tf}] (Aborted or Rejected)")
            except Exception as e:
                log.error(f"❌ Error predicting {sym} [{tf}]: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run_early_cycle()
