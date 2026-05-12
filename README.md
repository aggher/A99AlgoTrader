# AlgoTrader Institutional v2 — High-Frequency Ensemble Intelligence

An institutional-grade algorithmic trading platform leveraging a **50-model ensemble architecture** (XGBoost, LightGBM, Random Forest, LSTM, Transformer) and a **14-step high-integrity decision pipeline**.

## 🚀 Institutional Capabilities

*   **Ensemble Jury Voting**: Aggregated consensus from 5 heterogeneous models per symbol/timeframe.
*   **Meta-Model Filtering**: Secondary validation layer trained on ensemble error patterns to filter out low-confidence signals.
*   **125-Feature Vector**: Advanced feature engineering including Microstructure, Session Liquidity, MTF Confluence, and Hurst Exponents.
*   **Multi-Source Validation (MSV)**: Real-time data quality control (DQC) with outlier detection and gap alignment.
*   **Glassmorphism Dashboard**: Premium real-time visualization with sub-second WebSocket updates and interactive equity curve auditing.

## 📊 Core Institutional Pairs

| Category | Symbols |
| :--- | :--- |
| **Major FX** | EUR/USD · GBP/USD · USD/JPY · AUD/USD · USD/CAD |
| **Crosses** | EUR/JPY · GBP/AUD · EUR/AUD · GBP/NZD |
| **Metals** | XAU/USD (Gold) |

---

## 🛠️ Quick Start (Cloud Deployment)

### 1. Provision Infrastructure
We recommend **Fly.io** for the backend (persistent SQLite volumes) and **Vercel** for the frontend.

### 2. Configure Production Secrets
Use the provided helper scripts to securely synchronize your API keys:
```bash
./setup_production_secrets.sh
```

### 3. Deploy Backend (Fly.io)
```bash
fly launch
fly volumes create trading_data --size 10
fly deploy
```

### 4. Deploy Frontend (Vercel)
Connect your repository to Vercel and set `VITE_API_URL` to your Fly.io app address.

---

## 🏗️ Technical Architecture

```
trading-platform/
├── backend/
│   ├── institutional_orchestrator.py # 14-Step Decision Tree
│   ├── model_ensemble/              # XGB, LGBM, RF, LSTM, Transformer
│   ├── feature_engine/              # 125-Feature Microstructure Vector
│   ├── data_pipeline/               # DQC + Multi-Source Validation
│   ├── regime_engine/               # Volatility & Trend Regime Detector
│   ├── prediction_engine.py         # Real-time Signal Aggregator
│   └── dashboard_api.py             # FastAPI + WebSockets
├── frontend/
│   ├── vercel.json                  # SPA Routing & Proxy
│   └── src/components/              # Glassmorphism UI Components
├── fly.toml                         # Fly.io Deployment Config
├── Dockerfile                       # TA-Lib Optimized Container
└── production_sanity_check.py       # 10-Point Readiness Auditor
```

## 🛡️ Risk & Safety Gates
Signals are only approved for execution when they pass the **AlphaScoringEngine**:
*   **Directional Probability** > 65%
*   **Meta-Model Confidence** > 0.60
*   **Regime Alignment**: No "UNKNOWN" or high-volatility choppy regimes allowed.
*   **Liquidity Check**: Minimal volume/spread impact validation.

---

## ⚖️ License
Proprietary Institutional Codebase. For research and algorithmic validation purposes only.
