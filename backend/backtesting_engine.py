"""
backtesting_engine.py — High-integrity historical ensemble simulation.
Synchronized with the 14-step Institutional Decision Tree.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from backend.data_collector import get_ohlcv, SYMBOL_MAP, ALL_SYMBOLS
from backend.institutional_orchestrator import InstitutionalOrchestrator
from backend.database import init_db

log = logging.getLogger(__name__)

# Institutional Costs & Constraints
CAPITAL0    = 10_000.0
RISK_PCT    = 0.01  # 1% per trade
COMMISSION  = 5.0   # $5 flat fee per execution

@dataclass
class BacktestResult:
    symbol:        str
    timeframe:     str
    total_trades:  int   = 0
    wins:          int   = 0
    losses:        int   = 0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio:  float = 0.0
    max_drawdown:  float = 0.0
    total_return:  float = 0.0
    equity_curve:  List[float] = field(default_factory=list)
    trade_log:     List[dict]  = field(default_factory=list)

    def summary(self) -> str:
        return (f"\n{'─'*50}\n"
                f"INSTITUTIONAL BACKTEST: {self.symbol} [{self.timeframe}]\n"
                f"Trades    : {self.total_trades}  WinRate: {self.win_rate:.1%}\n"
                f"PF        : {self.profit_factor:.2f}  Sharpe: {self.sharpe_ratio:.2f}\n"
                f"MaxDD     : {self.max_drawdown:.1%}  Return: {self.total_return:.1%}\n"
                f"{'─'*50}")

def run_backtest(
    symbol: str,
    timeframe: str,
    prob: float = 0.65,
    limit: int = 1000
) -> Optional[BacktestResult]:
    """Runs a backtest using the full 14-step Orchestrator logic."""
    log.info(f"🚀 Running Synchronized Backtest for {symbol} [{timeframe}]...")
    
    # 1. Load Data
    df = get_ohlcv(symbol, timeframe, limit=limit)
    if len(df) < 200:
        log.warning(f"Insufficient data for {symbol} ({len(df)} rows)")
        return None

    orchestrator = InstitutionalOrchestrator(SYMBOL_MAP)
    capital = CAPITAL0
    equity  = [capital]
    trades  = []
    gross_w = gross_l = 0.0

    # OPTIMIZATION: Pre-calculate all 120 features once
    log.info("Generating features for the entire historical period...")
    feat_all = orchestrator.feature_engine.generate_features(df)
    
    # Minimum bars required for Baseline (PSI needs 550 rows if available)
    start_idx = 600 if len(feat_all) > 800 else 200
    
    # Simulating Bar-by-Bar "Real-time" snapshots
    i = start_idx
    while i < len(feat_all) - 1:
        # Current "Live" views (slices of the pre-calculated data)
        snapshot = df.iloc[:i+1] 
        feat_snapshot = feat_all.iloc[:i+1]
        
        # DECISION TREE EVALUATION
        res = orchestrator.process_snapshot(symbol, timeframe, snapshot, feat_df=feat_snapshot)
        
        # Enforce prob override
        if res.get("status") == "COMPLETED":
            if res.get("probability", 0.0) < prob:
                res["status"] = "REJECTED_BY_BACKTEST_PROB_THRESHOLD"

        if res.get("status") == "COMPLETED":
            lbl = res["signal"]
            entry_price = res["fill_price"]
            sl = res.get("stop_loss", entry_price * 0.995 if lbl == "BUY" else entry_price * 1.005)
            tp = res.get("take_profit", entry_price * 1.01 if lbl == "BUY" else entry_price * 0.99)
            
            # Realistic SL/TP Simulation
            exit_idx = i + 1
            exit_price = df['close'].iloc[i+1] # Default to next bar if nothing hit
            reason = "TIME_EXIT"
            
            # Check up to 10 bars forward for SL/TP hits
            lookahead = min(i + 10, len(df) - 1)
            for j in range(i + 1, lookahead + 1):
                high = df['high'].iloc[j]
                low  = df['low'].iloc[j]
                
                if lbl == "BUY":
                    if low <= sl:
                        exit_price = sl
                        exit_idx = j
                        reason = "STOP_LOSS"
                        break
                    if high >= tp:
                        exit_price = tp
                        exit_idx = j
                        reason = "TAKE_PROFIT"
                        break
                else: # SELL
                    if high >= sl:
                        exit_price = sl
                        exit_idx = j
                        reason = "STOP_LOSS"
                        break
                    if low <= tp:
                        exit_price = tp
                        exit_idx = j
                        reason = "TAKE_PROFIT"
                        break
                
                # If we reach the end of lookahead without hitting SL/TP
                if j == lookahead:
                    exit_price = df['close'].iloc[j]
                    exit_idx = j
            
            # PnL Calculation
            pnl_pct = (exit_price - entry_price) / entry_price if lbl == "BUY" else (entry_price - exit_price) / entry_price
            
            # Dynamic Position Sizing (Risk 1% of equity based on SL distance)
            sl_dist = abs(entry_price - sl) / entry_price
            if sl_dist < 0.0001: sl_dist = 0.005 # Cap min risk distance
            
            position_size = (capital * RISK_PCT) / sl_dist
            pnl_val = (position_size * pnl_pct) - (COMMISSION * 2) # Entry + Exit commissions
            
            capital += pnl_val
            if pnl_val > 0: gross_w += pnl_val
            else: gross_l += abs(pnl_val)
            
            trades.append({
                "time": str(df.index[i]),
                "signal": lbl,
                "entry": round(entry_price, 5),
                "exit": round(exit_price, 5),
                "pnl": round(pnl_val, 2),
                "reason": reason,
                "confidence": round(res.get("probability", 0.0), 2)
            })
            equity.append(capital)
            
            # Skip to exit to avoid overlapping trades
            i = exit_idx + 1
        else:
            i += 1

    n = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    eq_s = pd.Series(equity)
    rets = eq_s.pct_change().dropna()
    sha  = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    
    return BacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        total_trades=n,
        wins=wins,
        losses=n-wins,
        win_rate=wins/n if n else 0.0,
        profit_factor=gross_w / (gross_l + 1e-9),
        sharpe_ratio=sha,
        max_drawdown=float(((eq_s - eq_s.cummax()) / eq_s.cummax()).min()),
        total_return=(capital - CAPITAL0) / CAPITAL0,
        equity_curve=[round(v, 2) for v in equity],
        trade_log=trades
    )

def run_all_backtests(timeframes: List[str] = ["1h"]):
    init_db()
    symbols = ["EURUSD", "AUDUSD", "GBPUSD"]
    for sym in symbols:
        for tf in timeframes:
            res = run_backtest(sym, tf)
            if res:
                print(res.summary())

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_all_backtests()
