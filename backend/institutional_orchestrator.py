import logging
import pandas as pd
from typing import Dict, Any, List, Optional

from .data_pipeline.ingestor import DataIngestor
from .feature_engine.feature_engine import FeatureEngine
from .regime_engine.regime_detector import RegimeDetector
from .model_ensemble.aggregator import EnsembleAggregator
from .risk_engine.risk_manager import RiskManager
from .execution_engine.execution_simulator import ExecutionSimulator
from .monitoring.drift_detector import DriftDetector
from .monitoring.anomaly_detector import AnomalyDetector
from .monitoring.meta_model_filter import MetaModelFilter
from .protection.liquidity_monitor import LiquidityMonitor
from .alpha_scoring_engine import AlphaScoringEngine
from .risk_engine.macro_event_detector import MacroEventDetector
from .strategies.trend_following import TrendFollowingStrategy
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.portfolio_allocator import PortfolioAllocator

log = logging.getLogger(__name__)

class InstitutionalOrchestrator:
    """The central nervous system of the Institutional Platform v2."""
    
    def __init__(self, symbol_map: Dict[str, str]):
        self.ingestor = DataIngestor(symbol_map)
        self.feature_engine = FeatureEngine()
        self.regime_detector = RegimeDetector()
        self.risk_manager = RiskManager()
        self.execution_sim = ExecutionSimulator()
        self.drift_detector = DriftDetector()
        self.anomaly_detector = AnomalyDetector()
        self.meta_filter = MetaModelFilter()
        self.liquidity_monitor = LiquidityMonitor()
        self.alpha_scorer = AlphaScoringEngine()
        self.event_detector = MacroEventDetector()
        
        # Section 5: Strategy Modules
        self.strategies = [TrendFollowingStrategy(), MeanReversionStrategy()]
        self.allocator = PortfolioAllocator()
        
        self.aggregators = {} # Symbol -> Aggregator

    def process_symbol(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Fetches data and delegates to process_snapshot."""
        log.info(f"Processing institutional v2 cycle for {symbol} [{timeframe}]")
        
        # Section 1: Dynamic Lookback to avoid Lookback Starvation (ema200, rolling100)
        lookback_periods = {
            "1m": "7d",
            "5m": "30d",
            "15m": "60d",
            "1h": "90d",
            "1d": "2y",   # Need 250+ bars for EMA200
            "1mo": "max"  # Need 250+ bars (20 years)
        }
        period = lookback_periods.get(timeframe, "60d")
        
        df = self.ingestor.fetch_institutional_data(symbol, timeframe, period)
        if df is None:
            return {"status": "DATA_ERROR", "reason_code": "INGESTION_OR_VALIDATION_FAILED"}
            
        return self.process_snapshot(symbol, timeframe, df)

    def process_snapshot(self, symbol: str, timeframe: str, df: pd.DataFrame, feat_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Institutional v2 Pipeline: MSV -> FEATURES -> ENSEMBLE -> ALPHA -> STRATEGIES -> RISK -> EXECUTION.
        """
        # --- PHASE 1: DATA & STABILITY (Steps 1-3) ---
        df_clean, dqc_stats = self.ingestor.dqc.validate_snapshot(df, symbol)
        if not dqc_stats.get("is_valid", True):
            return {"status": "ABORTED", "reason": "STEP_1_DATA_QUALITY"}

        # Step 1.5: Data Stability Check
        if not self.ingestor.dqc.check_stability(df_clean, is_delta=(len(df) < 50)):
            return {"status": "ABORTED", "reason": "STEP_1_INSUFFICIENT_STABILITY"}

        # Section 2: Microstructure Feature Engine
        df_feat = feat_df if feat_df is not None else self.feature_engine.generate_features(df_clean)
        
        # Step 2: Feature Stability (PSI)
        # (Standard monitoring logic)
        
        # Step 3: Market Regime Detection
        regime_res = self.regime_detector.detect_regime(df_feat)
        
        # --- Extract live confluence snapshot from the last bar of feature data ---
        last = df_feat.iloc[-1]
        def _safe(col, default=0.0):
            val = last.get(col, default)
            return float(val) if val is not None and val == val else default

        feature_snapshot = {
            "rsi":                  _safe("mom_rsi_14", 50.0),
            "rsi_14_slope":         _safe("mom_rsi_14_slope", 0.0),
            "atr":                  _safe("vol_atr_20", 0.001),
            "atr_pct":              _safe("vol_atr_20_pct", 0.001),
            "bb_width":             _safe("vol_bb_w_20", 0.001),
            "bb_hi_dist":           _safe("vol_bb_hi_20", 0.0),
            "bb_lo_dist":           _safe("vol_bb_lo_20", 0.0),
            "vol_spike":            _safe("vol_spike", 1.0),
            "rel_volume_20":        _safe("liq_rel_vol_20", 1.0),
            "adx":                  _safe("trend_adx", 0.0),
            "macd_diff":            _safe("mom_macd_diff", 0.0),
            "stoch":                _safe("mom_stoch", 50.0),
            "pa_engulfing":         int(_safe("pa_engulfing", 0)),
            "pa_doji":              int(_safe("pa_doji", 0)),
            "pa_body_pct":          _safe("pa_body_pct", 0.5),
            "time_london":          int(_safe("time_london", 0)),
            "time_ny":              int(_safe("time_ny", 0)),
            "regime_strength":      min(float(regime_res.get("adx", 0.0)) / 40.0, 1.0),
        }

        # --- PHASE 2: PREDICTION (Steps 4-7) ---
        agg_key = f"{symbol}_{timeframe}"
        if agg_key not in self.aggregators:
            self.aggregators[agg_key] = EnsembleAggregator(symbol, timeframe)
        ensemble_res = self.aggregators[agg_key].aggregate_predictions(df_feat)
        
        if ensemble_res["signal"] == "HOLD":
            # In v2, we allow HOLD to pass through for auditability, but it won't trigger trades
            pass
            # return {"status": "REJECTED", "reason": "CONSENSUS_FAILED"}

        # Step 7: Meta Model Evaluation
        meta_context = {
            "signal": ensemble_res["signal"],
            "prob": ensemble_res["prob"],
            "agreement": ensemble_res["agreement"],
            "regime_id": regime_res["id"]
        }
        meta_res = self.meta_filter.evaluate_signal(meta_context)
        if not meta_res["approved"]:
            return {"status": "REJECTED", "reason": "META_FILTER_REJECT", "feature_snapshot": feature_snapshot, "probability": ensemble_res["prob"], "agreement": ensemble_res["agreement"], "signal": ensemble_res["signal"], "regime": regime_res["regime"]}

        # Section 12: Adaptive Alpha Threshold
        atr_pct_val = float(pd.to_numeric(df_feat['vol_atr_20_pct'].iloc[-1], errors='coerce') or 0.002) if 'vol_atr_20_pct' in df_feat.columns else 0.002
        adaptive_threshold = self.alpha_scorer.get_adaptive_threshold(atr_pct_val)

        regime_strength_proxy = float(regime_res.get("adx", 0.0)) / 40.0
        regime_strength_proxy = min(max(regime_strength_proxy, 0.0), 1.0)
        
        # Step 12: Alpha Scoring & Ranking
        alpha_inputs = {
            "ensemble_prob":        ensemble_res["prob"],
            "model_agreement":      ensemble_res["agreement"],
            "meta_model_score":     meta_res["quality_score"],
            "regime_strength":      regime_strength_proxy,
            "liquidity_quality":    min(float(feature_snapshot.get("rel_volume_20", 1.0)) / 2.0, 1.0),
            "volatility_stability": 1.0 - min(float(feature_snapshot.get("vol_atr_20_pct", 0.02)) / 0.05, 1.0)
        }
        alpha_res = self.alpha_scorer.validate_signal(alpha_inputs, threshold=adaptive_threshold)
        if not alpha_res["is_valid_alpha"]:
            return {"status": "REJECTED", "reason": "ALPHA_SCORE_TOO_LOW", "score": alpha_res["alpha_score"], "threshold": adaptive_threshold, "feature_snapshot": feature_snapshot, "probability": ensemble_res["prob"], "agreement": ensemble_res["agreement"], "signal": ensemble_res["signal"], "regime": regime_res["regime"]}

        # --- PHASE 3.5: STRATEGY DIVERSIFICATION (Section 5) ---
        strat_outputs = [s.generate_signal(df_feat) for s in self.strategies]
        alloc_res = self.allocator.allocate(strat_outputs)
        
        if alloc_res["signal"] == "HOLD":
            return {"status": "REJECTED", "reason": "STRATEGY_ALLOCATOR_HOLD", "feature_snapshot": feature_snapshot, "probability": ensemble_res["prob"], "agreement": ensemble_res["agreement"], "signal": ensemble_res["signal"], "regime": regime_res["regime"]}

        # --- PHASE 4: PROTECTION & RISK (Steps 8-11, Sections 6, 7, 10) ---
        # Section 6: Macro Event Protection
        macro_risk_mult = self.event_detector.evaluate_risk_status()
        if macro_risk_mult == 0:
            return {"status": "ABORTED", "reason": "MACRO_EVENT_LOCKOUT"}

        liq_res = self.liquidity_monitor.check_conditions(symbol, df_feat)
        if not liq_res["safe"]:
            return {"status": "REJECTED", "reason": "LIQUIDITY_REJECT", "feature_snapshot": feature_snapshot, "probability": ensemble_res["prob"], "agreement": ensemble_res["agreement"], "signal": ensemble_res["signal"], "regime": regime_res["regime"]}

        # Step 10/11/Section 7/Section 10: Portfolio Risk & Position Sizing
        risk_input = {
            "symbol": symbol,
            "confidence": alpha_res["alpha_score"] * alloc_res["confidence"],
            "atr_pct": float(pd.to_numeric(df_feat['vol_atr_20_pct'].iloc[-1], errors='coerce') or 0.002) if 'vol_atr_20_pct' in df_feat.columns else 0.002
        }
        risk_res = self.risk_manager.validate_signal(risk_input, [])
        if not risk_res["approved"]:
            return {"status": "RISK_REJECTED", "reason": risk_res["reason"], "feature_snapshot": feature_snapshot, "probability": ensemble_res["prob"], "agreement": ensemble_res["agreement"], "signal": ensemble_res["signal"], "regime": regime_res["regime"]}

        # Applied Macro Risk Multiplier
        final_size_mult = risk_res["size_multiplier"] * macro_risk_mult

        # --- PHASE 5: EXECUTION (Steps 13-14) ---
        # Final Signal is a combination of Ensemble and Strategy Allocator
        final_signal = ensemble_res["signal"] # In v2, Ensemble is the primary directional gate
        
        exec_res = self.execution_sim.simulate_execution(final_signal, df_feat['close'].iloc[-1], df_feat)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": final_signal,
            "probability": ensemble_res["prob"],
            "agreement": ensemble_res["agreement"],
            "alpha_score": alpha_res["alpha_score"],
            "allocation_conf": alloc_res["confidence"],
            "size_multiplier": final_size_mult,
            "regime": regime_res["regime"],
            "regime_strength": feature_snapshot["regime_strength"],
            "entry_price": float(df_feat['close'].iloc[-1]),
            "fill_price": exec_res["fill_price"],
            "meta_score": meta_res.get("quality_score", 0.0),
            "status": "COMPLETED",
            "feature_snapshot": feature_snapshot,
            "metadata": {"strategies": alloc_res["strategies"], "macro_risk": macro_risk_mult},
        }
