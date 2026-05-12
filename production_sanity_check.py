import os
import sys
import logging
import sqlite3
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("SanityCheck")

def check_step(name, condition):
    if condition:
        log.info(f"✅ {name}: PASSED")
        return True
    else:
        log.error(f"❌ {name}: FAILED")
        return False

def run_sanity_check():
    log.info("--- Institutional Trading System: Production Sanity Check ---")
    results = []

    # 1. Directory Structure
    results.append(check_step("Models Directory", os.path.exists("models/ensemble")))
    results.append(check_step("Data Directory", os.path.exists("data")))
    
    # 2. Database Integrity
    db_path = "data/trading.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT count(*) FROM ohlcv")
            row_count = c.fetchone()[0]
            results.append(check_step(f"Database Connectivity (Rows: {row_count})", row_count > 0))
            
            c.execute("SELECT count(distinct symbol) FROM ohlcv")
            sym_count = c.fetchone()[0]
            results.append(check_step(f"Asset Diversity (Symbols: {sym_count})", sym_count >= 10))
            conn.close()
        except Exception as e:
            results.append(check_step(f"Database Query: {e}", False))
    else:
        results.append(check_step("Database File Found", False))

    # 3. Model Training Status
    ensemble_count = 0
    for root, dirs, files in os.walk("models/ensemble"):
        ensemble_count += len([f for f in files if f.endswith('.joblib')])
    results.append(check_step(f"Ensemble Model Check ({ensemble_count}/50 targets)", ensemble_count >= 40))

    # 4. Orchestrator Cold-Start Test
    try:
        from backend.institutional_orchestrator import InstitutionalOrchestrator
        from backend.data_collector import SYMBOL_MAP
        orch = InstitutionalOrchestrator(SYMBOL_MAP)
        results.append(check_step("Orchestrator Initialization", True))
        
        # Test a dummy prediction (Step 1-14)
        dummy_df = pd.DataFrame({
            "open": [1.1]*300, "high": [1.11]*300, "low": [1.09]*300, "close": [1.1]*300, "volume": [1000]*300
        }, index=pd.date_range(datetime.now(), periods=300, freq='H'))
        res = orch.process_snapshot("EURUSD", "1h", dummy_df)
        results.append(check_step("Decision Pipeline Execution", res.get("status") in ["COMPLETED", "ABORTED"]))
    except Exception as e:
        results.append(check_step(f"Orchestrator Logic: {e}", False))

    # 5. Deployment Configs
    results.append(check_step("Dockerfile Presence", os.path.exists("Dockerfile")))
    results.append(check_step("Fly.io Config (fly.toml)", os.path.exists("fly.toml")))
    results.append(check_step("Vercel Config (vercel.json)", os.path.exists("frontend/vercel.json")))

    log.info("--- Final Verdict ---")
    if all(results):
        log.info("🚀 SYSTEM STATUS: PRODUCTION READY")
    else:
        log.warning("⚠️ SYSTEM STATUS: ACTION REQUIRED (Check Failed Items)")

if __name__ == "__main__":
    run_sanity_check()
