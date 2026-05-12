"""
train_xgboost_model.py — XGBoost training pipeline with walk-forward validation.

For each symbol × timeframe:
  1. Load OHLCV (DB first, yfinance fallback)
  2. Compute features & labels
  3. Walk-forward cross-validation (5 folds)
  4. Time-series train/test split (80/20)
  5. GridSearchCV hyperparameter tuning
  6. Final model evaluation (accuracy, precision, recall, F1, Sharpe, drawdown)
  7. Save model to models/{symbol}_{tf}.json
  8. Store metrics in DB
"""
from __future__ import annotations
import json, logging, os, warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

from backend.data_collector import get_ohlcv, collect_all, SYMBOL_MAP, TIMEFRAME_CONFIG, ALL_SYMBOLS
from backend.feature_engineering import compute_features, label_data, FEATURE_COLUMNS, merge_mtf_features
from backend.database import SessionLocal, ModelMetric, init_db

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_TIMEFRAMES = ["1mo", "1d", "1h", "15m", "5m"]
LABEL_THRESHOLD    = float(os.getenv("LABEL_THRESHOLD_PCT", "0.4"))
FORWARD_CANDLES    = int(os.getenv("LABEL_FORWARD_CANDLES", "10"))
MIN_ROWS           = 200
TRAIN_RATIO        = 0.80
WF_FOLDS           = 5
MIN_CLASS_SAMPLES  = 10   # minimum BUY/SELL rows required to train

# Shorter timeframes have noisier labels → use a lower threshold to produce
# more BUY/SELL examples and avoid near-all-HOLD datasets.
TF_LABEL_THRESHOLDS = {
    "1mo": 2.0,
    "1d":  1.0,
    "1h":  0.40,
    "15m": 0.20,
    "5m":  0.15,
    "1m":  0.10,
}

PARAM_GRID = {
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.05, 0.1],
    "n_estimators":     [100, 200],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma":            [0, 0.1],
}

CLASS_NAMES  = ["BUY", "HOLD", "SELL"]

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder().fit(CLASS_NAMES)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_data(symbol: str, tf: str) -> pd.DataFrame:
    df = get_ohlcv(symbol, tf, limit=2000)
    if len(df) >= MIN_ROWS:
        return df
    log.info("  DB empty for %s/%s — downloading…", symbol, tf)
    import yfinance as yf
    ticker = SYMBOL_MAP[symbol]
    interval, period = TIMEFRAME_CONFIG[tf]
    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                               "Close": "close", "Volume": "volume"})
    raw = raw[["open","high","low","close","volume"]].dropna()
    raw.index = pd.to_datetime(raw.index, utc=True).tz_localize(None)
    return raw


def _sharpe(returns: pd.Series, ann: int = 252) -> float:
    s = returns.std()
    return float(returns.mean() / s * np.sqrt(ann)) if s > 0 else 0.0


def _max_dd(equity: pd.Series) -> float:
    pk = equity.cummax()
    return float(((equity - pk) / pk.replace(0, np.nan)).min())


def _sim_returns(df_test: pd.DataFrame, preds: np.ndarray,
                 spread: float = 0.0002) -> pd.Series:
    close = df_test["close"].values
    ret   = np.zeros(len(preds))
    for i in range(len(preds) - 1):
        lbl = CLASS_NAMES[preds[i]]
        bar = (close[i+1] - close[i]) / close[i]
        if lbl == "BUY":
            ret[i] = bar - spread
        elif lbl == "SELL":
            ret[i] = -bar - spread
    return pd.Series(ret)


def _walk_forward(X: pd.DataFrame, y: np.ndarray, n: int) -> List[dict]:
    fold_size = len(X) // (n + 1)
    results   = []
    for fold in range(1, n + 1):
        te_s = fold * fold_size
        te_e = te_s + fold_size
        X_tr, y_tr = X.iloc[:te_s], y[:te_s]
        X_te, y_te = X.iloc[te_s:te_e], y[te_s:te_e]
        if len(X_tr) < 50 or len(X_te) < 10:
            continue
        # Skip fold if training split lacks all 3 classes
        uniq_tr = np.unique(y_tr)
        if len(uniq_tr) < 3:
            log.info("    WF %d/%d skipped (only %d classes in train)", fold, n, len(uniq_tr))
            continue
        m = xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                               use_label_encoder=False, eval_metric="mlogloss",
                               n_estimators=100, max_depth=4, learning_rate=0.1,
                               n_jobs=-1, random_state=42)
        m.fit(X_tr, y_tr, verbose=False)
        p   = m.predict(X_te)
        acc = accuracy_score(y_te, p)
        f1  = f1_score(y_te, p, average="weighted", zero_division=0)
        results.append({"fold": fold, "accuracy": acc, "f1": f1})
        log.info("    WF %d/%d acc=%.3f f1=%.3f", fold, n, acc, f1)
    return results


# ── Main training ─────────────────────────────────────────────────────────────

def train(
    symbols: List[str] = None,
    timeframes: List[str] = None,
    tune: bool = True,
) -> None:
    print("\n" + "!"*80)
    print("WARNING: train_xgboost_model.py is LEGACY and out of sync with Institutional v2.")
    print("Please use 'python -m backend.institutional_train' for the full 125-feature ensemble.")
    print("!"*80 + "\n")
    
    init_db()
    symbols    = symbols    or ALL_SYMBOLS
    timeframes = timeframes or DEFAULT_TIMEFRAMES

    for sym in symbols:
        for tf in timeframes:
            log.info("▶ %s [%s]", sym, tf)
            try:
                _train_one(sym, tf, tune)
            except Exception as e:
                log.error("  FAILED %s/%s: %s", sym, tf, e)


# ── Training-specific SQLite engine (NullPool avoids lock contention with API) ─
def _get_train_session():
    from backend.database import DB_PATH
    eng = create_engine(f"sqlite:///{DB_PATH}",
                        connect_args={"check_same_thread": False},
                        poolclass=NullPool)
    SM  = sessionmaker(bind=eng, autocommit=False, autoflush=False)
    return SM()


def _train_one(symbol: str, tf: str, tune: bool) -> None:
    raw = _load_data(symbol, tf)
    if len(raw) < MIN_ROWS:
        log.warning("  Skip %s/%s — not enough data (%d rows)", symbol, tf, len(raw))
        return

    # ── Feature Engineering ───────────────────────────────────────────────────
    df = compute_features(raw)
    
    # ── MTF Enrichment ────────────────────────────────────────────────────────
    mtf_cols = []
    
    # Mapping of which HTFs to use for each base timeframe
    MTF_HIERARCHY = {
        "5m":  ["15m", "1h"],
        "15m": ["1h", "1d"],
        "1h":  ["1d", "1mo"],
        "1d":  ["1mo"]
    }

    if tf in MTF_HIERARCHY:
        for htf in MTF_HIERARCHY[tf]:
            h_raw = _load_data(symbol, htf)
            if not h_raw.empty:
                h_feat = compute_features(h_raw)
                df = merge_mtf_features(df, h_feat, htf)
                for col in ["ema_trend", "rsi", "ema_50", "ema_200"]:
                    mtf_cols.append(f"{col}_{htf}")

    # Use timeframe-specific label threshold to avoid all-HOLD datasets
    thr = TF_LABEL_THRESHOLDS.get(tf, LABEL_THRESHOLD)
    labels = label_data(df, thr, FORWARD_CANDLES)

    current_features = FEATURE_COLUMNS + mtf_cols
    valid  = df[current_features].notna().all(axis=1)
    df     = df[valid]
    labels = labels[valid]
    # remove tail where labels are unreliable
    df     = df.iloc[:-FORWARD_CANDLES]
    labels = labels.iloc[:-FORWARD_CANDLES]

    if len(df) < MIN_ROWS:
        log.warning("  Skip %s/%s — too few valid rows", symbol, tf)
        return

    X = df[current_features]
    y = LE.transform(labels)

    counts = dict(zip(*np.unique(y, return_counts=True)))
    log.info("  Rows=%d  Labels=%s  threshold=%.2f%%", len(X),
             {k: int(v) for k, v in counts.items()}, thr)

    # Skip if BUY or SELL has too few samples for meaningful training
    non_hold = [(k, v) for k, v in counts.items() if k != 1]  # 1 = HOLD
    if any(v < MIN_CLASS_SAMPLES for _, v in non_hold):
        log.warning("  Skip %s/%s — insufficient BUY/SELL samples (need >= %d). "
                    "Try lower LABEL_THRESHOLD_PCT.", symbol, tf, MIN_CLASS_SAMPLES)
        return

    # Walk-forward CV
    wf = _walk_forward(X, y, WF_FOLDS)
    if wf:
        log.info("  WF mean acc=%.3f f1=%.3f",
                 np.mean([r["accuracy"] for r in wf]),
                 np.mean([r["f1"] for r in wf]))

    # Train/test split
    split   = int(len(X) * TRAIN_RATIO)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split],      y[split:]

    # Need all 3 classes in training set for multi-class XGBoost
    if len(np.unique(y_tr)) < 3:
        log.warning("  Skip %s/%s — training split lacks all 3 classes", symbol, tf)
        return

    # Hyperparameter tuning
    if tune:
        log.info("  GridSearchCV…")
        gs = GridSearchCV(
            xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                               use_label_encoder=False, eval_metric="mlogloss",
                               n_jobs=-1, random_state=42),
            PARAM_GRID, cv=3, scoring="f1_weighted", verbose=0, n_jobs=-1,
        )
        gs.fit(X_tr, y_tr)
        best = gs.best_params_
        log.info("  Best params: %s", best)
    else:
        best = {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 200,
                "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0.1}

    # Final model
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, use_label_encoder=False,
        eval_metric="mlogloss", n_jobs=-1, random_state=42, **best,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    # Evaluation
    preds = model.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    prec  = precision_score(y_te, preds, average="weighted", zero_division=0)
    rec   = recall_score(y_te, preds, average="weighted", zero_division=0)
    f1    = f1_score(y_te, preds, average="weighted", zero_division=0)

    sim_ret = _sim_returns(df.iloc[split:], preds)
    eq      = (1 + sim_ret).cumprod()
    sharpe  = _sharpe(sim_ret)
    dd      = _max_dd(eq)
    gross_w = sim_ret[sim_ret > 0].sum()
    gross_l = abs(sim_ret[sim_ret < 0].sum())
    pf      = min(float(gross_w / gross_l), 100.0) if gross_l > 0 else 100.0

    log.info("  ✓ acc=%.3f prec=%.3f rec=%.3f f1=%.3f sharpe=%.2f dd=%.2f%% pf=%.2f",
             acc, prec, rec, f1, sharpe, dd*100, pf)
    
    # Dynamically filter target names for classification_report
    present_labels = sorted(np.unique(np.concatenate([y_te, preds])))
    present_names  = [CLASS_NAMES[i] for i in present_labels]
    log.info("\n%s", classification_report(y_te, preds,
             labels=present_labels, target_names=present_names, zero_division=0))

    path = MODELS_DIR / f"{symbol}_{tf}.json"
    model.save_model(str(path))
    log.info("  Saved → %s", path)

    _save_metrics(symbol, tf, acc, prec, rec, f1, sharpe, dd, pf, best)


def _save_metrics(symbol, tf, acc, prec, rec, f1, sharpe, dd, pf, params):
    # Use NullPool engine so we don't fight the running API server for the DB lock
    s = _get_train_session()
    try:
        from backend.database import ModelMetric as _MM
        s.add(_MM(symbol=symbol, timeframe=tf, accuracy=acc,
                  precision=prec, recall=rec, f1_score=f1,
                  sharpe_ratio=sharpe, max_drawdown=dd,
                  profit_factor=pf, params=json.dumps(params)))
        s.commit()
        log.info("  Metrics saved to DB ✓")
    except Exception as e:
        s.rollback(); log.error("Metrics save failed: %s", e)
    finally:
        s.close()


def load_model(symbol: str, tf: str) -> Optional[xgb.XGBClassifier]:
    p = MODELS_DIR / f"{symbol}_{tf}.json"
    if not p.exists():
        return None
    m = xgb.XGBClassifier()
    m.load_model(str(p))
    return m


if __name__ == "__main__":
    train(tune=False)
