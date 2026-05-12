"""
volatility_breakout_detector.py — ATR + Bollinger Band breakout detection.

volatility_breakout = 1 when price closes OUTSIDE Bollinger Bands
                        AND ATR > rolling ATR average.

breakout_direction  = +1 (upside), -1 (downside), 0 (none)
"""
from __future__ import annotations
import pandas as pd
import ta


def detect_volatility_breakout(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    bb_window: int = 20,
    bb_std: float = 2.0,
    atr_window: int = 14,
    atr_avg: int = 5,
) -> pd.Series:
    bb    = ta.volatility.BollingerBands(close, bb_window, bb_std)
    atr   = ta.volatility.AverageTrueRange(high, low, close, atr_window).average_true_range()
    atr_m = atr.shift(1).rolling(atr_avg, min_periods=1).mean()
    outside = (close > bb.bollinger_hband()) | (close < bb.bollinger_lband())
    atr_increasing = atr > atr.shift(1)
    return (outside & atr_increasing).astype(int).rename("volatility_breakout")


def breakout_direction(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    bb_window: int = 20,
    bb_std: float = 2.0,
    atr_window: int = 14,
    atr_avg: int = 5,
) -> pd.Series:
    bb  = ta.volatility.BollingerBands(close, bb_window, bb_std)
    atr = ta.volatility.AverageTrueRange(high, low, close, atr_window).average_true_range()
    atr_m = atr.shift(1).rolling(atr_avg, min_periods=1).mean()
    exp   = atr > atr_m
    d     = pd.Series(0, index=close.index, dtype=int)
    d[exp & (close > bb.bollinger_hband())] =  1
    d[exp & (close < bb.bollinger_lband())] = -1
    return d.rename("breakout_direction")
