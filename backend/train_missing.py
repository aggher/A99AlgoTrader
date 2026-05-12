"""
train_missing.py — Surgically train only missing models.
"""
import logging
from backend.database import SessionLocal, ModelMetric, init_db
from backend.data_collector import ALL_SYMBOLS, TIMEFRAME_CONFIG
from backend.train_xgboost_model import _train_one, init_db as train_init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

def run():
    train_init_db()
    s = SessionLocal()
    try:
        metrics = s.query(ModelMetric.symbol, ModelMetric.timeframe).all()
        existing = set((m.symbol, m.timeframe) for m in metrics)
    finally:
        s.close()

    # We skip 1m because it's usually excluded or manually handled
    tfs = [tf for tf in TIMEFRAME_CONFIG.keys() if tf != "1m"]
    
    missing = []
    for sym in ALL_SYMBOLS:
        for tf in tfs:
            if (sym, tf) not in existing:
                missing.append((sym, tf))

    if not missing:
        log.info("No missing models to train (excluding 1m).")
        return

    log.info("Found %d missing models. Starting surgical training...", len(missing))
    
    for sym, tf in missing:
        log.info("▶ STARTING MISSING: %s [%s]", sym, tf)
        try:
            # tune=True to match standard training pipeline
            _train_one(sym, tf, tune=True)
            log.info("✓ COMPLETED MISSING: %s [%s]", sym, tf)
        except Exception as e:
            log.error("❌ FAILED MISSING: %s/%s: %s", sym, tf, e)

if __name__ == "__main__":
    run()
