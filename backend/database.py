"""
database.py — SQLite schema (upgradeable to PostgreSQL/TimescaleDB).
Tables: ohlcv, signals, trades, model_metrics
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column, DateTime, Float, Index, Integer, String, Text,
    create_engine, event,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH      = DATA_DIR / "trading.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

@event.listens_for(engine, "connect")
def _set_pragma(conn, _):
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.close()

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class OHLCV(Base):
    __tablename__ = "ohlcv"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    symbol    = Column(String(20), nullable=False)
    timeframe = Column(String(5),  nullable=False)
    timestamp = Column(DateTime,   nullable=False)
    open      = Column(Float,      nullable=False)
    high      = Column(Float,      nullable=False)
    low       = Column(Float,      nullable=False)
    close     = Column(Float,      nullable=False)
    volume    = Column(Float,      nullable=False)
    __table_args__ = (
        Index("ix_ohlcv_sym_tf_ts", "symbol", "timeframe", "timestamp", unique=True),
    )


class Signal(Base):
    __tablename__ = "signals"
    id                  = Column(Integer, primary_key=True, autoincrement=True)
    symbol              = Column(String(20), nullable=False)
    timeframe           = Column(String(5),  nullable=False)
    timestamp           = Column(DateTime,   nullable=False, default=datetime.utcnow)
    signal              = Column(String(5),  nullable=False)
    prob_buy            = Column(Float,      nullable=True)
    prob_sell           = Column(Float,      nullable=True)
    prob_hold           = Column(Float,      nullable=True)
    
    # Layer 2: Meta-Model
    confidence          = Column(Float,      nullable=True) # Step 12 Formula Score
    meta_score          = Column(Float,      nullable=True)
    agreement           = Column(Float,      nullable=True) # Ensemble Consensus Score
    meta_decision       = Column(String(10), nullable=True) # ACCEPT/REJECT
    
    # Layer 3: Regime & Context
    market_regime       = Column(String(20), nullable=True) # TRENDING/RANGING
    regime_strength     = Column(Float,      nullable=True)
    
    entry               = Column(Float,      nullable=True)
    stop_loss           = Column(Float,      nullable=True)
    take_profit         = Column(Float,      nullable=True)
    
    # Context Data
    rsi_divergence      = Column(Integer,    nullable=True)
    volume_spike        = Column(Integer,    nullable=True)
    volatility_breakout = Column(Integer,    nullable=True)
    position_size       = Column(Float,      nullable=True)
    technical_metadata  = Column(Text,       nullable=True) # JSON blob for extra technicals
    __table_args__ = (Index("ix_signals_sym_ts", "symbol", "timestamp"),)


class Trade(Base):
    __tablename__ = "trades"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol      = Column(String(20), nullable=False)
    timeframe   = Column(String(5),  nullable=False)
    direction   = Column(String(5),  nullable=False)
    entry_price = Column(Float,      nullable=False)
    exit_price  = Column(Float,      nullable=True)
    stop_loss   = Column(Float,      nullable=True)
    take_profit = Column(Float,      nullable=True)
    entry_time  = Column(DateTime,   nullable=False)
    exit_time   = Column(DateTime,   nullable=True)
    pnl         = Column(Float,      nullable=True)
    status      = Column(String(10), nullable=False, default="open")


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    symbol       = Column(String(20), nullable=False)
    timeframe    = Column(String(5),  nullable=False)
    trained_at   = Column(DateTime,   nullable=False, default=datetime.utcnow)
    accuracy     = Column(Float,      nullable=True)
    precision    = Column(Float,      nullable=True)
    recall       = Column(Float,      nullable=True)
    f1_score     = Column(Float,      nullable=True)
    sharpe_ratio = Column(Float,      nullable=True)
    max_drawdown = Column(Float,      nullable=True)
    profit_factor= Column(Float,      nullable=True)
    params       = Column(Text,       nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("DB initialised at", DB_PATH)
