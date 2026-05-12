"""
data_collector.py — Fetches OHLCV data from Yahoo Finance.

Symbols (10 Core Institutional Pairs):
  Forex  : EURUSD, GBPUSD, GBPAUD, EURAUD, GBPNZD, USDJPY, AUDUSD, USDCAD, EURJPY
  Metals : XAUUSD
Timeframes: 1m, 5m, 15m, 1h, 1d, 1mo
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from .data_pipeline.ingestor import DataIngestor
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.database import SessionLocal, OHLCV, init_db

log = logging.getLogger(__name__)

# ── Symbol registry  (logical name → Yahoo Finance ticker) ────────────────────
SYMBOL_MAP: Dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "GBPAUD": "GBPAUD=X",
    "EURAUD": "EURAUD=X",
    "GBPNZD": "GBPNZD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "EURJPY": "EURJPY=X",
    "XAUUSD": "GC=F",   # Gold Futures
}

ALL_SYMBOLS: List[str] = list(SYMBOL_MAP.keys())

# Timeframe → (yf interval, initial lookback period)
TIMEFRAME_CONFIG: Dict[str, Tuple[str, str]] = {
    "1m":  ("1m",  "7d"),
    "5m":  ("5m",  "30d"),
    "15m": ("15m", "60d"),
    "1h":  ("1h",  "180d"),
    "1d":  ("1d",  "max"),
    "1mo": ("1mo", "max"),
}


# Initialize institutional components
ingestor = DataIngestor(SYMBOL_MAP)

def _fetch_institutional(symbol: str, timeframe: str, initial: bool) -> Optional[pd.DataFrame]:
    period = TIMEFRAME_CONFIG[timeframe][1] if initial else "2d"
    df = ingestor.fetch_institutional_data(symbol, timeframe, period)
    if df is not None:
        # Normalize for DB table consistency
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "timestamp"
    return df


def store_ohlcv(symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    session = SessionLocal()
    inserted = 0
    try:
        data_to_insert = []
        for ts, row in df.iterrows():
            data_to_insert.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": ts.to_pydatetime(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume)
            })
        
        if data_to_insert:
            stmt = sqlite_insert(OHLCV).on_conflict_do_nothing(
                index_elements=["symbol", "timeframe", "timestamp"]
            )
            result = session.execute(stmt, data_to_insert)
            try:
                inserted = result.rowcount
            except AttributeError:
                inserted = len(data_to_insert) # Fallback to total attempted
        session.commit()
    except Exception as e:
        session.rollback()
        log.error("DB insert error %s/%s: %s", symbol, timeframe, e)
    finally:
        session.close()
    return inserted


def collect_all(initial: bool = False) -> None:
    log.info("=== Institutional Data Sync (initial=%s) ===", initial)
    for symbol in ALL_SYMBOLS:
        for tf in TIMEFRAME_CONFIG.keys():
            df = _fetch_institutional(symbol, tf, initial)
            if df is not None:
                n = store_ohlcv(symbol, tf, df)
                log.info("  %s [%s] \u2705 Validated \u2192 %d rows", symbol, tf, n)
            time.sleep(0.1)
    log.info("=== Sync Complete ===")


def get_ohlcv(symbol: str, timeframe: str, limit: int = 600) -> pd.DataFrame:
    session = SessionLocal()
    try:
        rows = (
            session.query(OHLCV)
            .filter(OHLCV.symbol == symbol, OHLCV.timeframe == timeframe)
            .order_by(OHLCV.timestamp.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()
    if not rows:
        return pd.DataFrame()
    data = [{"timestamp": r.timestamp, "open": r.open, "high": r.high,
              "low": r.low, "close": r.close, "volume": r.volume}
            for r in reversed(rows)]
    df = pd.DataFrame(data).set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    collect_all(initial=True)
