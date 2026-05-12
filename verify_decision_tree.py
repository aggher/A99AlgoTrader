import logging
import pandas as pd
import numpy as np
from backend.institutional_orchestrator import InstitutionalOrchestrator

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def test_decision_tree():
    log.info("🧪 Starting Decision Tree Verification Trace...")
    
    symbol_map = {"EURUSD": "EURUSD=X"}
    orchestrator = InstitutionalOrchestrator(symbol_map)
    
    # Run a prediction cycle for EURUSD [15m]
    symbol = "EURUSD"
    tf = "15m"
    
    log.info(f"\n--- TRACING: {symbol} [{tf}] ---")
    result = orchestrator.process_symbol(symbol, tf)
    
    # 14-Step Trace Mapping
    log.info("\nDECISION TREE STATUS:")
    if result["status"] == "ABORTED":
        log.warning(f"❌ ABORTED at {result['reason']} (Reason: {result.get('details', 'N/A')})")
    elif result["status"] == "REJECTED":
        log.warning(f"⚠️ REJECTED at {result['reason']} (Score/Prob: {result.get('score', result.get('prob', 'N/A'))})")
    elif result["status"] == "RISK_REJECTED":
        log.warning(f"🛑 RISK REJECTED: {result['reason']}")
    elif result["status"] == "COMPLETED":
        log.info("✅ SIGNAL GENERATED SUCCESSFULLY")
        log.info(f"   Direction: {result['signal']}")
        log.info(f"   Confidence: {result.get('alpha_score', 0.0):.4f}")
        log.info(f"   Agreement: {result.get('agreement', 0.0):.2f}")
        log.info(f"   Regime: {result['regime']}")
        log.info(f"   Fill Price: {result['fill_price']}")
    else:
        log.error(f"❓ UNKNOWN STATUS: {result['status']}")

if __name__ == "__main__":
    test_decision_tree()
