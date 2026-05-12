"""
volume_spike_detector.py — Abnormal volume spike detection.

volume_spike  = 1 when volume > rolling_mean × multiplier, else 0
volume_ratio  = volume / rolling_mean  (continuous strength feature)
"""
from __future__ import annotations
import pandas as pd


def detect_volume_spikes(
    volume: pd.Series,
    window: int = 20,
    multiplier: float = 2.0,
) -> pd.Series:
    mean = volume.rolling(window, min_periods=max(1, window // 2)).mean()
    return (volume > mean * multiplier).astype(int).rename("volume_spike")


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    mean = volume.rolling(window, min_periods=max(1, window // 2)).mean()
    return (volume / mean.replace(0, float("nan"))).fillna(1.0).rename("volume_ratio")
