import logging
import time
import pandas as pd
from backend.institutional_orchestrator import InstitutionalOrchestrator
from backend.data_collector import SYMBOL_MAP

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def deep_trace():
    orch = InstitutionalOrchestrator(SYMBOL_MAP)
    pairs = ["EURUSD", "AUDUSD", "XAUUSD"]
    timeframes = ["1h", "15m", "5m"]
    
    trace_data = []
    
    log.info("🕵️ Starting Institutional Deep Trace Audit...")
    
    for sym in pairs:
        for tf in timeframes:
            log.info(f"\nAUDIT: {sym} [{tf}]")
            res = orch.process_symbol(sym, tf)
            
            status = res.get("status")
            reason = res.get("reason", "N/A")
            
            trace_entry = {
                "asset": sym,
                "tf": tf,
                "status": status,
                "reason": reason,
                "prob": res.get("probability", 0.0),
                "agreement": res.get("agreement", 0.0),
                "confidence": res.get("confidence", 0.0)
            }
            trace_data.append(trace_entry)
            
            if status == "COMPLETED":
                log.info(f"✅ PASSED: {res['signal']} | Prob: {res['probability']:.4f} | Agree: {res['agreement']:.2f}")
            elif status == "ABORTED":
                log.warning(f"❌ ABORTED: {reason} | Details: {res.get('max_psi', res.get('details', 'N/A'))}")
            elif status == "REJECTED":
                log.warning(f"⚠️ REJECTED: {reason} | Prob: {res.get('probability', 0):.4f} | Agree: {res.get('agreement', 0):.2f}")
            elif status == "RISK_REJECTED":
                log.warning(f"🛑 RISK: {reason}")
            else:
                log.error(f"❓ OTHER: {status}")
            
            time.sleep(1)

    df_trace = pd.DataFrame(trace_data)
    log.info("\n--- FINAL AUDIT SUMMARY ---")
    log.info(df_trace.to_string())

if __name__ == "__main__":
    deep_trace()
