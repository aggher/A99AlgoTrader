"""
dashboard_api.py — FastAPI REST + WebSocket server.

Endpoints:
  GET  /api/prices          — latest price per symbol
  GET  /api/ohlcv/{symbol}  — OHLCV bars for chart rendering
  GET  /api/signals         — latest signals from DB
  GET  /api/performance     — model metrics
  GET  /api/backtest/{sym}  — backtest equity curve + trade log
  GET  /api/symbols         — all tracked symbols
  POST /api/train           — background model training
  POST /api/collect         — background data collection
  GET  /health
  WS   /ws                  — real-time price + alert push
"""
from __future__ import annotations
import asyncio, json, logging, os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()
log = logging.getLogger(__name__)

from backend.database import SessionLocal, init_db, OHLCV, Signal, ModelMetric, get_db
from backend.data_collector import ALL_SYMBOLS, get_ohlcv
from backend.prediction_engine import get_latest_signals, run_all
from backend.backtesting_engine import run_backtest
from backend.alert_service import register_ws_listener, unregister_ws_listener
from sqlalchemy.orm import Session


# ── App ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    asyncio.create_task(_price_broadcast())
    yield

app = FastAPI(title="AlgoTrader AI", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── WebSocket manager ──────────────────────────────────────────────────────────
class WSManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept(); self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections: self.connections.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.connections:
            try: await ws.send_json(data)
            except Exception: dead.append(ws)
        for ws in dead: self.disconnect(ws)

mgr = WSManager()


async def _price_broadcast():
    while True:
        try:
            await mgr.broadcast({"type": "prices", "data": _snapshot()})
        except Exception: pass
        await asyncio.sleep(5)


def _snapshot() -> list:
    s = SessionLocal()
    out = []
    try:
        for sym in ALL_SYMBOLS:
            row = (s.query(OHLCV)
                   .filter(OHLCV.symbol == sym, OHLCV.timeframe == "1h")
                   .order_by(OHLCV.timestamp.desc()).first())
            if row:
                out.append({"symbol": row.symbol, "close": row.close,
                             "open": row.open, "high": row.high,
                             "low": row.low, "volume": row.volume,
                             "timestamp": row.timestamp.isoformat()})
    finally: s.close()
    return out


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/api/prices")
def prices(): return {"prices": _snapshot()}


@app.get("/api/ohlcv/{symbol}")
def ohlcv(symbol: str, timeframe: str = "1h", limit: int = 300):
    df = get_ohlcv(symbol.upper(), timeframe, limit)
    if df.empty: return {"bars": []}
    bars = [{"time": int(ts.timestamp()), "open": r.open,
              "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
            for ts, r in df.iterrows()]
    return {"symbol": symbol, "timeframe": timeframe, "bars": bars}


@app.get("/api/signals")
def signals(limit: int = 60): return {"signals": get_latest_signals(limit)}


@app.get("/api/signals/refresh")
def refresh_signals(bg: BackgroundTasks):
    bg.add_task(run_all); return {"status": "queued"}


@app.get("/api/performance")
def performance(db: Session = Depends(get_db)):
    rows = db.query(ModelMetric).order_by(ModelMetric.trained_at.desc()).limit(200).all()
    return {"metrics": [{
        "symbol": r.symbol, "timeframe": r.timeframe,
        "trained_at": r.trained_at.isoformat(),
        "accuracy": r.accuracy, "precision": r.precision,
        "recall": r.recall, "f1_score": r.f1_score,
        "sharpe_ratio": r.sharpe_ratio, "max_drawdown": r.max_drawdown,
        "profit_factor": min(r.profit_factor, 100.0) if r.profit_factor else None,
        "params": json.loads(r.params) if r.params else {},
    } for r in rows]}


@app.get("/api/backtest/{symbol}")
def backtest(symbol: str, timeframe: str = "1h", prob: float = 0.60):
    r = run_backtest(symbol.upper(), timeframe, prob)
    if not r:
        return JSONResponse(status_code=404,
                            content={"error": "No data. Train model first."})
    return {"symbol": r.symbol, "timeframe": r.timeframe,
            "total_trades": r.total_trades, "wins": r.wins, "losses": r.losses,
            "win_rate": r.win_rate, "profit_factor": r.profit_factor,
            "sharpe_ratio": r.sharpe_ratio, "max_drawdown": r.max_drawdown,
            "total_return": r.total_return, "equity_curve": r.equity_curve,
            "trade_log": r.trade_log[-200:]}


@app.get("/api/symbols")
def symbols(): return {"symbols": ALL_SYMBOLS}


@app.post("/api/train")
def train_model(bg: BackgroundTasks, symbols: str = "", timeframes: str = "1h"):
    def _run():
        from backend.institutional_train import institutional_retrain
        # symbols parameter ignored in institutional_retrain for now, syncs all
        institutional_retrain()
    bg.add_task(_run)
    return {"status": "training queued"}


@app.post("/api/collect")
def collect_data(bg: BackgroundTasks):
    from backend.data_collector import collect_all
    bg.add_task(collect_all, True)
    return {"status": "collection queued"}


@app.get("/health")
def health(): return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await mgr.connect(ws)
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    register_ws_listener(q)
    await ws.send_json({"type": "symbols", "data": ALL_SYMBOLS})
    try:
        while True:
            try:
                alert = q.get_nowait()
                await ws.send_json(alert)
            except asyncio.QueueEmpty: pass
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=1.0)
                if msg == "ping": await ws.send_text("pong")
            except asyncio.TimeoutError: pass
    except WebSocketDisconnect: pass
    finally:
        unregister_ws_listener(q); mgr.disconnect(ws)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run("backend.dashboard_api:app",
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", "8000")), reload=True)
