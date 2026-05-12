"""
feature_engineering.py — Computes all technical features for XGBoost.

Feature groups:
  Trend      : EMA50, EMA200, slope, EMA spread, price_vs_ema50/200
  Momentum   : RSI, MACD, price momentum
  Volatility : ATR, BB width, volatility expansion
  Volume     : spike, ratio, volume_change_pct
  Structure  : candle body size, wick ratios, support/resistance distance
  Signals    : rsi_divergence, volume_spike, volatility_breakout
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from backend.rsi_divergence_detector import detect_rsi_divergence
from backend.volume_spike_detector import detect_volume_spikes, volume_ratio
from backend.volatility_breakout_detector import detect_volatility_breakout, breakout_direction

log = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    # Trend (6)
    "ema_20", "ema_50", "ema_100", "ema_200", "ema_trend", "ema_slope",
    # Momentum (7)
    "rsi", "macd", "macd_signal", "macd_hist", "momentum", "roc", "rsi_slope",
    # Volatility (6)
    "atr", "atr_change", "bollinger_width", "volatility_breakout", "range_expansion", "vol_ratio",
    # Volume (6)
    "volume_ma", "volume_spike", "volume_slope", "volume_change_rate", "volume_imbalance", "volume",
    # Structure (6)
    "support_distance", "resistance_distance", "sr_breakout", "pullback_depth", "swing_high", "swing_low",
    # Price Action (5)
    "candle_body", "upper_wick", "lower_wick", "doji", "engulfing",
    # Time Features (4)
    "hour", "session_asian", "session_london", "session_newyork",
    # Signals (1)
    "rsi_divergence",
    # Context (1)
    "price_change",
    # Note: MTF features will append _15m or _1h suffixes
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : OHLCV DataFrame (open, high, low, close, volume)
    Output: DataFrame enriched with the 25-feature dataset.
    """
    if df.empty or len(df) < 55:
        return df
    df = df.copy()

    # Ensure column names are lowercase
    df.columns = [c.lower() for c in df.columns]

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"].replace(0, np.nan).ffill().fillna(0)

    # ── Trend Indicators ───────────────────────────────────────────────────────
    df["ema_20"]          = EMAIndicator(close, 20).ema_indicator()
    df["ema_50"]          = EMAIndicator(close, 50).ema_indicator()
    df["ema_100"]         = EMAIndicator(close, 100).ema_indicator()
    df["ema_200"]         = EMAIndicator(close, 200).ema_indicator()
    df["ema_trend"]       = ((df["ema_50"] > df["ema_200"]) & (df["ema_20"] > df["ema_50"])).astype(int)
    df["ema_slope"]       = df["ema_50"].pct_change(5).fillna(0) # 5-bar slope for stability

    # ── Momentum Indicators ────────────────────────────────────────────────────
    df["rsi"]         = RSIIndicator(close, 14).rsi()
    df["rsi_slope"]   = df["rsi"].diff(3).fillna(0)
    _macd             = MACD(close)
    df["macd"]        = _macd.macd()
    df["macd_signal"] = _macd.macd_signal()
    df["macd_hist"]   = _macd.macd_diff()
    df["momentum"]    = (close - close.shift(1)) / close.shift(1).replace(0, np.nan)
    df["roc"]         = close.pct_change(10).fillna(0)

    # ── Volatility Indicators ──────────────────────────────────────────────────
    df["atr"]             = AverageTrueRange(high, low, close, 14).average_true_range()
    df["atr_change"]      = df["atr"].pct_change(5).fillna(0)
    _bb                   = BollingerBands(close, 20, 2)
    bb_mid                = _bb.bollinger_mavg()
    df["bollinger_width"] = (_bb.bollinger_hband() - _bb.bollinger_lband()) / bb_mid.replace(0, np.nan)
    df["volatility_breakout"] = detect_volatility_breakout(close, high, low)
    df["range_expansion"]     = (high - low) / (high - low).rolling(10).mean().replace(0, np.nan)
    df["vol_ratio"]           = df["atr"] / close.replace(0, np.nan)

    # ── Volume Features ───────────────────────────────────────────────────────
    df["volume_ma"]          = volume.rolling(20).mean()
    df["volume_spike"]       = detect_volume_spikes(volume, multiplier=1.5)
    df["volume_slope"]       = df["volume_ma"].pct_change(5).fillna(0)
    df["volume_change_rate"] = volume.pct_change().fillna(0)
    # Institutional Proxy: Volume/Price Imbalance
    df["volume_imbalance"]   = (volume * (close - low)) / ((high - low).replace(0, np.nan) * volume.mean())

    # ── Market Structure ───────────────────────────────────────────────────────
    rolling_high = high.rolling(50).max()
    rolling_low  = low.rolling(50).min()
    df["resistance_distance"] = (rolling_high - close) / close.replace(0, np.nan)
    df["support_distance"]    = (close - rolling_low) / close.replace(0, np.nan)
    
    df["sr_breakout"] = 0
    df.loc[close > rolling_high.shift(1), "sr_breakout"] = 1
    df.loc[close < rolling_low.shift(1), "sr_breakout"] = -1

    # Swing Levels
    df["swing_high"] = (high == rolling_high).astype(int)
    df["swing_low"]  = (low == rolling_low).astype(int)
    df["pullback_depth"] = (rolling_high - close) / (rolling_high - rolling_low).replace(0, np.nan)

    # ── Price Action ──────────────────────────────────────────────────────────
    candle_range     = (high - low).replace(0, np.nan)
    df["candle_body"]= ((close - df["open"]).abs() / candle_range).fillna(0)
    df["upper_wick"] = ((high - pd.concat([close, df["open"]], axis=1).max(axis=1)) / candle_range).fillna(0)
    df["lower_wick"] = ((pd.concat([close, df["open"]], axis=1).min(axis=1) - low) / candle_range).fillna(0)
    
    # Doji Detection (Body < 10% of Range)
    df["doji"] = (df["candle_body"] < 0.1).astype(int)
    # Engulfing (Simple)
    prev_body = (df["close"].shift(1) - df["open"].shift(1)).abs()
    curr_body = (close - df["open"]).abs()
    df["engulfing"] = ((curr_body > prev_body) & (df["candle_body"] > 0.6)).astype(int)

    # ── Time Features ─────────────────────────────────────────────────────────
    # We assume index is DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df["hour"] = df.index.hour
        # Professional Session Filters (UTC)
        df["session_asian"]   = df["hour"].between(0, 8).astype(int)
        df["session_london"]  = df["hour"].between(8, 16).astype(int)
        df["session_newyork"] = df["hour"].between(13, 21).astype(int)
    else:
        df["hour"] = 0
        df["session_asian"] = 0
        df["session_london"] = 0
        df["session_newyork"] = 0

    # ── Signals & Context ─────────────────────────────────────────────────────
    df["rsi_divergence"] = detect_rsi_divergence(close, df["rsi"])
    df["price_change"]   = close.pct_change().fillna(0)

    # Final cleanup
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return df


def merge_mtf_features(base_df: pd.DataFrame, higher_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Aligns higher timeframe indicators with lower timeframe base data."""
    if higher_df.empty:
        return base_df
    
    # Pick relevant MTF features: trend and momentum
    mtf_cols = ["ema_trend", "rsi", "ema_50", "ema_200"]
    available = [c for c in mtf_cols if c in higher_df.columns]
    
    subset = higher_df[available].copy()
    subset.columns = [f"{c}_{suffix}" for c in subset.columns]
    
    # Forward-fill merge
    merged = pd.merge_asof(base_df, subset, left_index=True, right_index=True)
    return merged.ffill().fillna(0)


def label_data(
    df: pd.DataFrame,
    threshold_pct: float = 0.4,
    forward_candles: int = 10,
) -> pd.Series:
    """BUY / SELL / HOLD labels via forward price movement."""
    thr          = threshold_pct / 100.0
    future_max   = df["close"].shift(-1).rolling(forward_candles, min_periods=1).max()
    future_min   = df["close"].shift(-1).rolling(forward_candles, min_periods=1).min()
    pct_up       = (future_max - df["close"]) / df["close"]
    pct_down     = (df["close"] - future_min) / df["close"]

    labels       = pd.Series("HOLD", index=df.index)
    labels[pct_up   >= thr] = "BUY"
    labels[pct_down >= thr] = "SELL"

    conflict = (pct_up >= thr) & (pct_down >= thr)
    labels[conflict & (pct_up >= pct_down)] = "BUY"
    labels[conflict & (pct_up <  pct_down)] = "SELL"
    return labels
