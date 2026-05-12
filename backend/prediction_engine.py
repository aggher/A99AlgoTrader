"""
prediction_engine.py — Real-time BUY/SELL/HOLD signal generation.

For each symbol × timeframe:
  1. Load latest OHLCV
  2. Compute features
  3. XGBoost predict → label + probability
  4. Risk management (entry/SL/TP/position size)
  5. Signal confluence check (signals must be ≥2 of 3 active for strong signal)
  6. Persist to DB
  7. Return SignalResult
"""
from __future__ import annotations
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

import numpy as np
import xgboost as xgb

from backend.data_collector import ALL_SYMBOLS, SYMBOL_MAP
from backend.institutional_orchestrator import InstitutionalOrchestrator
from backend.database import SessionLocal, Signal

log = logging.getLogger(__name__)

PROB_THRESHOLD = float(__import__("os").getenv("SIGNAL_PROBABILITY_THRESHOLD", "0.75"))


_ORCHESTRATOR: Optional[InstitutionalOrchestrator] = None

def get_orchestrator() -> InstitutionalOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = InstitutionalOrchestrator(SYMBOL_MAP)
    return _ORCHESTRATOR


@dataclass
class SignalResult:
    symbol:              str
    timeframe:           str
    timestamp:           str
    signal:              str
    probability:         float
    entry:               Optional[float]
    stop_loss:           Optional[float]
    take_profit:         Optional[float]
    position_size:       Optional[float]
    rsi_divergence:      int
    volume_spike:        int
    volatility_breakout: int
    confluence_score:    int    # 0-3: how many signals align
    rsi:                 float
    atr:                 float
    bb_width:            float
    
    # Institutional Layer Metadata
    prob_buy:            float
    prob_sell:           float
    prob_hold:           float
    meta_score:          float  # Layer 2 Score
    meta_decision:       str    # ACCEPT/REJECT
    market_regime:       str    # TRENDING/RANGING (Layer 3)
    regime_strength:     float
    agreement:           float  # Ensemble Consensus Score
    confidence:          float  # Step 12 Institutional Score

    def to_dict(self) -> dict:
        return asdict(self)

    def is_actionable(self) -> bool:
        # Enforce 3-Layer Institutional Rule
        return (self.signal != "HOLD"
                and self.probability >= PROB_THRESHOLD
                and self.meta_score >= 0.65  # Layer 2 Threshold
                and self.meta_decision == "ACCEPT"
                and self.market_regime != "UNKNOWN")

    def summary(self) -> str:
        return (f"Symbol: {self.symbol} [{self.timeframe}]\n"
                f"Signal: {self.signal}  Prob: {self.probability:.2f}"
                f"  Confluence: {self.confluence_score}/3\n"
                f"Entry: {self.entry}  SL: {self.stop_loss}  TP: {self.take_profit}")


def predict(symbol: str, timeframe: str) -> Optional[SignalResult]:
    orch = get_orchestrator()
    res = orch.process_symbol(symbol, timeframe)
    
    fs = res.get("feature_snapshot", {})
    sig = res.get("signal", "HOLD")

    # --- Derive confluence signals from live features ---
    rsi_val        = fs.get("rsi", 50.0)
    rsi_slope      = fs.get("rsi_14_slope", 0.0)
    atr_val        = fs.get("atr", 0.001)
    bb_width_val   = fs.get("bb_width", 0.001)
    vol_spike_val  = fs.get("vol_spike", 1.0)
    rel_vol        = fs.get("rel_volume_20", 1.0)
    bb_hi_dist     = fs.get("bb_hi_dist", 0.0)
    bb_lo_dist     = fs.get("bb_lo_dist", 0.0)
    adx_val        = fs.get("adx", 0.0)

    # RSI Divergence: slope moving against signal direction
    rsi_divergence = 0
    if sig == "BUY" and rsi_slope < -1.0:    
        rsi_divergence = -1
    elif sig == "SELL" and rsi_slope > 1.0:   
        rsi_divergence = 1

    # Volume Spike: current candle range > 1.5x prior 20-bar median
    volume_spike = 1 if vol_spike_val > 1.5 else 0

    # Volatility Breakout: price pierced a Bollinger Band
    volatility_breakout = 1 if (bb_hi_dist > 0 or bb_lo_dist < -0.001) else 0

    # Confluence Score: count of active signals (0-5 scale)
    confluence_signals = [
        rsi_divergence != 0,                          
        volume_spike == 1,                            
        volatility_breakout == 1,                     
        rel_vol > 1.2,                               
        adx_val > 25,                                
    ]
    confluence_score = sum(confluence_signals)

    if res["status"] != "COMPLETED":
        log.warning(f"Institutional skip for {symbol} [{timeframe}]: {res['status']}")
        # Create a dummy HOLD result to show in the dashboard
        result = SignalResult(
            symbol=symbol, timeframe=timeframe, timestamp=datetime.utcnow().isoformat(),
            signal="HOLD", probability=res.get("probability", 0.5), 
            entry=None, stop_loss=None, take_profit=None,
            position_size=0.0, rsi_divergence=rsi_divergence, volume_spike=volume_spike, volatility_breakout=volatility_breakout,
            confluence_score=confluence_score, rsi=round(rsi_val, 2), atr=round(atr_val, 6), bb_width=round(bb_width_val, 6),
            prob_buy=0.0, prob_sell=0.0, prob_hold=1.0, meta_score=round(res.get("agreement", 0.0), 4),
            meta_decision="REJECT", market_regime=res.get("regime", "UNKNOWN"), regime_strength=0.0,
            agreement=round(res.get("agreement", 0.0), 4), confidence=0.0
        )
        _save(result)
        return result

    # Map to legacy SignalResult for DB/Frontend compatibility
    raw_prob = res["probability"]
    # Distribute remaining probability so that prob_buy + prob_sell + prob_hold == 1.0
    remainder = 1.0 - raw_prob
    if sig == "BUY":
        prob_buy  = raw_prob
        prob_sell = remainder * 0.33
        prob_hold = remainder * 0.67
    elif sig == "SELL":
        prob_sell = raw_prob
        prob_buy  = remainder * 0.33
        prob_hold = remainder * 0.67
    else:  # HOLD
        prob_hold = raw_prob
        prob_buy  = remainder * 0.5
        prob_sell = remainder * 0.5

    result = SignalResult(
        symbol              = symbol,
        timeframe           = timeframe,
        timestamp           = datetime.utcnow().isoformat(),
        signal              = sig,
        probability         = round(raw_prob, 4),
        entry               = res.get("fill_price") if sig != "HOLD" else None,
        stop_loss           = res.get("fill_price", 0) * (0.99 if sig == "BUY" else 1.01) if sig != "HOLD" else None,
        take_profit         = res.get("fill_price", 0) * (1.02 if sig == "BUY" else 0.98) if sig != "HOLD" else None,
        position_size       = round(res.get("size_multiplier", 1.0), 4),
        rsi_divergence      = rsi_divergence,
        volume_spike        = volume_spike,
        volatility_breakout = volatility_breakout,
        confluence_score    = confluence_score,
        rsi      = round(rsi_val, 2),
        atr      = round(atr_val, 6),
        bb_width = round(bb_width_val, 6),
        prob_buy  = round(prob_buy, 4),
        prob_sell = round(prob_sell, 4),
        prob_hold = round(prob_hold, 4),
        meta_score      = round(res.get("meta_score", res.get("agreement", 0.0)), 4),
        meta_decision   = "ACCEPT",
        market_regime   = res.get("regime", "UNKNOWN"),
        regime_strength = round(res.get("regime_strength", 0.0), 4),
        agreement       = round(res.get("agreement", 0.0), 4),
        confidence      = round(res.get("alpha_score", 0.0), 4),
    )

    _save(result)
    return result



def _save(r: SignalResult) -> None:
    def _sanitize(val):
        if isinstance(val, (float, np.floating)) and (not np.isfinite(val)):
            return 0.0
        return val

    technical_meta = json.dumps({
        "rsi":      round(float(_sanitize(r.rsi)),     2),
        "atr":      round(float(_sanitize(r.atr)),     6),
        "bb_width": round(float(_sanitize(r.bb_width)), 6),
        "confluence_score": r.confluence_score,
    })

    s = SessionLocal()
    try:
        s.add(Signal(
            symbol=r.symbol, timeframe=r.timeframe,
            timestamp=datetime.utcnow(), signal=r.signal,
            prob_buy=_sanitize(r.prob_buy),
            prob_sell=_sanitize(r.prob_sell),
            prob_hold=_sanitize(r.prob_hold),
            meta_score=_sanitize(r.meta_score),
            agreement=_sanitize(r.agreement),
            confidence=_sanitize(r.confidence),
            meta_decision=r.meta_decision,
            market_regime=r.market_regime,
            regime_strength=_sanitize(r.regime_strength),
            entry=_sanitize(r.entry),
            stop_loss=_sanitize(r.stop_loss),
            take_profit=_sanitize(r.take_profit),
            rsi_divergence=r.rsi_divergence,
            volume_spike=r.volume_spike,
            volatility_breakout=r.volatility_breakout,
            position_size=_sanitize(r.position_size),
            technical_metadata=technical_meta,
        ))
        s.commit()
    except Exception as e:
        s.rollback(); log.error("Signal save error: %s", e)
    finally:
        s.close()

# load_cached_model removed — model loading is handled by EnsembleAggregator/base_model.py


def run_all(timeframes: List[str] = None) -> List[SignalResult]:
    tfs = timeframes or ["1mo", "1d", "1h", "15m", "5m"]
    results = []
    for sym in ALL_SYMBOLS:
        for tf in tfs:
            r = predict(sym, tf)
            if r:
                results.append(r)
                if r.is_actionable():
                    log.info("🔔 %s", r.summary())
    return results


def get_latest_signals(limit: int = 60) -> List[dict]:
    s = SessionLocal()
    try:
        from sqlalchemy import func
        # Find latest timestamp for each (symbol, timeframe)
        subq = (s.query(Signal.symbol, Signal.timeframe, func.max(Signal.timestamp).label("max_ts"))
                .group_by(Signal.symbol, Signal.timeframe)
                .subquery())
        
        # Join back to get full records
        rows = (s.query(Signal)
                .join(subq, (Signal.symbol == subq.c.symbol) &
                            (Signal.timeframe == subq.c.timeframe) &
                            (Signal.timestamp == subq.c.max_ts))
                .order_by(Signal.timestamp.desc())
                .limit(limit).all())

        def _sanitize(val):
            if isinstance(val, float) and (not np.isfinite(val)):
                return 0.0
            return val

        result = []
        for r in rows:
            # Parse technical metadata if available
            tech = {}
            if r.technical_metadata:
                try:
                    tech = json.loads(r.technical_metadata)
                except Exception:
                    tech = {}

            result.append({
                "id": r.id, "symbol": r.symbol, "timeframe": r.timeframe,
                "timestamp": r.timestamp.isoformat(), "signal": r.signal,
                "probability": _sanitize(max(r.prob_buy or 0.0, r.prob_sell or 0.0, r.prob_hold or 0.0)),
                "prob_buy":  _sanitize(r.prob_buy  or 0.0),
                "prob_sell": _sanitize(r.prob_sell or 0.0),
                "prob_hold": _sanitize(r.prob_hold or 0.0),
                "entry": _sanitize(r.entry),
                "stop_loss": _sanitize(r.stop_loss), "take_profit": _sanitize(r.take_profit),
                "rsi_divergence": r.rsi_divergence, "volume_spike": r.volume_spike,
                "volatility_breakout": r.volatility_breakout,
                "position_size": _sanitize(r.position_size),
                "confluence_score": tech.get("confluence_score", 0),
                "rsi":      tech.get("rsi",      None),
                "atr":      tech.get("atr",      None),
                "bb_width": tech.get("bb_width", None),
                "meta_score": _sanitize(r.meta_score),
                "agreement":  _sanitize(r.agreement),
                "confidence": _sanitize(r.confidence),
                "meta_decision": r.meta_decision,
                "market_regime": r.market_regime,
                "regime_strength": _sanitize(r.regime_strength or 0.0),
            })
        return result
    finally:
        s.close()
