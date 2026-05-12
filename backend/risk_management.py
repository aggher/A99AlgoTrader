"""
risk_management.py — Position sizing, risk control, and portfolio exposure limits.

Implements:
  • Kelly-inspired position sizing (capped at 2% risk per trade)
  • ATR-based stop loss/take profit calculation
  • Max daily loss circuit breaker
  • Portfolio exposure limits
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

RISK_PER_TRADE   = float(os.getenv("RISK_PER_TRADE",   "0.01"))   # 1% default
MAX_DAILY_LOSS   = float(os.getenv("MAX_DAILY_LOSS",   "0.05"))   # 5%
INITIAL_CAPITAL  = float(os.getenv("INITIAL_CAPITAL",  "10000"))
MAX_EXPOSURE_PCT = 0.20   # max 20% of capital in open positions simultaneously
SL_ATR_MULT      = 1.5
TP_ATR_MULT      = 3.0
MAX_RISK_PCT     = 0.02   # cap risk at 2% regardless of Kelly


@dataclass
class RiskParameters:
    symbol:        str
    direction:     str          # BUY | SELL
    entry:         float
    stop_loss:     float
    take_profit:   float
    position_size: float        # units / lots
    risk_amount:   float        # $ at risk
    rr_ratio:      float        # reward : risk ratio


def calculate_levels(
    close:     float,
    atr:       float,
    direction: str,
) -> tuple[float, float, float]:
    """Return (entry, stop_loss, take_profit) for a given direction."""
    if atr <= 0:
        atr = close * 0.001   # fallback: 0.1% of price

    if direction == "BUY":
        entry = close
        sl    = round(entry - SL_ATR_MULT * atr, 6)
        tp    = round(entry + TP_ATR_MULT * atr, 6)
    else:
        entry = close
        sl    = round(entry + SL_ATR_MULT * atr, 6)
        tp    = round(entry - TP_ATR_MULT * atr, 6)
    return entry, sl, tp


def calculate_position_size(
    account_equity:   float,
    entry:            float,
    stop_loss:        float,
    risk_pct:         float = RISK_PER_TRADE,
    daily_pnl:        float = 0.0,
) -> float:
    """
    Calculate position size in units.

    Uses fixed-fraction risk management:
      units = (equity × risk_pct) / |entry − stop_loss|

    Checks:
      • Caps risk at MAX_RISK_PCT
      • Returns 0 if daily loss limit breached
    """
    risk_pct = min(risk_pct, MAX_RISK_PCT)

    # Daily loss circuit breaker
    if daily_pnl < -account_equity * MAX_DAILY_LOSS:
        log.warning("Daily loss limit reached (%.2f). Trading halted.", daily_pnl)
        return 0.0

    sl_dist = abs(entry - stop_loss)
    if sl_dist == 0:
        return 0.0

    risk_amount = account_equity * risk_pct
    units       = risk_amount / sl_dist
    return round(units, 4)


def build_risk_params(
    symbol:         str,
    direction:      str,
    close:          float,
    atr:            float,
    account_equity: float = INITIAL_CAPITAL,
    daily_pnl:      float = 0.0,
) -> Optional[RiskParameters]:
    """Full risk parameter calculation for a proposed trade."""
    if direction == "HOLD":
        return None

    entry, sl, tp = calculate_levels(close, atr, direction)
    size          = calculate_position_size(account_equity, entry, sl,
                                            daily_pnl=daily_pnl)
    if size == 0:
        return None

    sl_dist    = abs(entry - sl)
    tp_dist    = abs(tp - entry)
    risk_amount= size * sl_dist
    rr         = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0

    return RiskParameters(
        symbol        = symbol,
        direction     = direction,
        entry         = round(entry, 6),
        stop_loss     = sl,
        take_profit   = tp,
        position_size = size,
        risk_amount   = round(risk_amount, 2),
        rr_ratio      = rr,
    )
