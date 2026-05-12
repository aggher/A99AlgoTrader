"""
rsi_divergence_detector.py — Bullish and bearish RSI divergence detection.

  +1  → bullish divergence (price lower low, RSI higher low)
  -1  → bearish divergence (price higher high, RSI lower high)
   0  → none
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _local_lows(s: pd.Series, order: int) -> pd.Series:
    mask = pd.Series(False, index=s.index)
    arr = s.values
    for i in range(order, len(arr) - order):
        if arr[i] == arr[i - order: i + order + 1].min():
            mask.iloc[i] = True
    return mask


def _local_highs(s: pd.Series, order: int) -> pd.Series:
    mask = pd.Series(False, index=s.index)
    arr = s.values
    for i in range(order, len(arr) - order):
        if arr[i] == arr[i - order: i + order + 1].max():
            mask.iloc[i] = True
    return mask


def detect_rsi_divergence(
    close: pd.Series,
    rsi: pd.Series,
    order: int = 5,
    lookback: int = 40,
) -> pd.Series:
    """Return per-bar RSI divergence signal: +1, -1, or 0."""
    divergence = pd.Series(0, index=close.index, dtype=int)

    pl = _local_lows(close, order)
    rl = _local_lows(rsi, order)
    ph = _local_highs(close, order)
    rh = _local_highs(rsi, order)

    idx = list(close.index)
    low_idx  = [t for t in idx if pl[t]]
    high_idx = [t for t in idx if ph[t]]
    c = close.values
    r = rsi.values

    for i in range(order, len(idx)):
        ts = idx[i]

        if pl.iloc[i] and rl.iloc[i]:
            prev = [t for t in low_idx if t < ts]
            if prev:
                pi = idx.index(prev[-1])
                if (i - pi) <= lookback and c[i] < c[pi] and r[i] > r[pi]:
                    divergence.iloc[i] = 1

        if ph.iloc[i] and rh.iloc[i]:
            prev = [t for t in high_idx if t < ts]
            if prev:
                pi = idx.index(prev[-1])
                if (i - pi) <= lookback and c[i] > c[pi] and r[i] < r[pi]:
                    divergence.iloc[i] = -1

    return divergence
