# AlgoTrader AI — XGBoost Trading Signal Platform

AI-powered Forex and stock trading signal platform targeting **65–75% signal accuracy** using XGBoost, walk-forward validation, and multi-signal confluence filtering.

## Symbols Tracked

| Group     | Symbols |
|-----------|---------|
| Forex     | EUR/USD · USD/JPY · XAU/USD |
| GBP pairs | GBP/USD · GBP/EUR · GBP/JPY · GBP/CHF · GBP/AUD · GBP/CAD · GBP/NZD |
| AUD pairs | AUD/USD · AUD/EUR · AUD/JPY · AUD/GBP · AUD/CAD · AUD/NZD · AUD/CHF |
| Stocks    | AAPL · MSFT · TSLA · NVDA |

## Quick Start

### 1. Install Python dependencies
```bash
cd trading-platform
pip install -r requirements.txt
```

### 2. Copy environment file
```bash
copy .env.example .env
```
*(Edit `.env` to add Telegram/email credentials if desired.)*

### 3. Collect initial market data
```bash
python -m backend.data_collector
```
This fetches historical OHLCV data from Yahoo Finance for all 21 symbols × 4 timeframes.

### 4. Train XGBoost models
```bash
python -m backend.train_xgboost_model
```
Training runs walk-forward cross-validation + GridSearchCV hyperparameter tuning.
Models are saved to `models/{SYMBOL}_{TF}.json`.

### 5. Start the FastAPI backend
```bash
uvicorn backend.dashboard_api:app --reload --port 8000
```

### 6. Install & start the frontend
```bash
cd frontend
npm install
npm run dev
```
Open **http://localhost:5173** in your browser.

### 7. (Optional) Start the automated pipeline
```bash
python -m backend.scheduler
```
Runs data collection + prediction every 60 seconds, with weekly Sunday 02:00 UTC retraining.

## Project Structure

```
trading-platform/
├── backend/
│   ├── database.py                  # SQLite schema (OHLCV, signals, trades, metrics)
│   ├── data_collector.py            # Yahoo Finance OHLCV fetcher
│   ├── feature_engineering.py       # 24 technical features across 6 groups
│   ├── rsi_divergence_detector.py   # Bullish/bearish RSI divergence
│   ├── volume_spike_detector.py     # Volume spike (binary + ratio)
│   ├── volatility_breakout_detector.py  # ATR + Bollinger Band breakout
│   ├── train_xgboost_model.py       # XGBoost training + walk-forward CV
│   ├── prediction_engine.py         # Real-time signal generation + confluence
│   ├── backtesting_engine.py        # Historical simulation with costs
│   ├── risk_management.py           # Position sizing, SL/TP, daily loss limit
│   ├── alert_service.py             # Telegram, email, WebSocket alerts
│   ├── dashboard_api.py             # FastAPI REST + WebSocket server
│   └── scheduler.py                 # APScheduler automation
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── LivePrices.tsx       # Price chip grid
│       │   ├── SignalCards.tsx      # Signal cards with confluence
│       │   ├── Chart.tsx            # lightweight-charts candlestick
│       │   ├── TradeHistory.tsx     # Signal history table
│       │   ├── PerformanceMetrics.tsx  # Metrics + equity curve
│       │   └── AlertPanel.tsx       # Real-time alert toasts
│       ├── hooks/useWebSocket.ts
│       └── types.ts
├── models/                          # Saved XGBoost models (.json)
├── data/                            # SQLite database
├── requirements.txt
└── .env.example
```

## Feature Engineering (27+ features)

| Group     | Features |
|-----------|----------|
| **Trend** | EMA 50, EMA 200, EMA Spread, Trend Slope, Price vs EMA 50/200 |
| **Momentum**| RSI (14), MACD, MACD Signal, MACD Histogram, Price Momentum (5-period) |
| **Volatility**| ATR (14), Bollinger Band Width, Volatility Expansion |
| **Volume** | Volume Spike, Volume Moving Average (20), Volume Ratio, Volume Change % |
| **Structure**| Support/Resistance Distance (50-period), Candle Body Size, Wick Ratios (Upper/Lower) |
| **Confluence**| RSI Divergence, Volume Spike Detection, Volatility Breakout Detection |

## Signal Filtering

Signals are only flagged **actionable** when:
- Model probability ≥ **70%**
- **Confluence score ≥ 2/3** (RSI Divergence + Volume Spike + Volatility Breakout aligning with the prediction)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/prices` | Latest price per symbol |
| GET | `/api/ohlcv/{symbol}` | OHLCV bars for chart |
| GET | `/api/signals` | Latest AI signals |
| GET | `/api/performance` | Model metrics |
| GET | `/api/backtest/{symbol}` | Backtest results + equity curve |
| POST | `/api/train` | Trigger model training |
| POST | `/api/collect` | Trigger data collection |
| WS  | `/ws` | Real-time prices + alerts |
