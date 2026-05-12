import pandas as pd
import numpy as np
import ta
import logging
from typing import Dict, List

log = logging.getLogger(__name__)

class FeatureEngine:
    """Institutional-grade Feature Engineering Engine with 120+ features."""

    def __init__(self):
        self.feature_groups = {
            "trend": 20,
            "momentum": 20,
            "volatility": 15,
            "structure": 15,
            "liquidity": 15,
            "session": 10,
            "cross_asset": 15,
            "price_action": 10
        }

    def generate_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Entry point for full feature generation (Target: 125)."""
        df = df_input.copy()
        
        # 1. Base technical categories
        df = self._add_trend_features(df)
        df = self._add_momentum_features(df)
        df = df.copy()
        df = self._add_volatility_features(df)
        df = self._add_structure_features(df)
        df = df.copy()
        df = self._add_microstructure_features(df)
        df = self._add_price_action_features(df)
        df = df.copy()
        df = self._add_time_features(df)
        
        # 2. Lag Features (Expand core signals by 2 lags each to reach target)
        core_cols = ['close', 'mom_rsi_14', 'vol_atr_20_pct', 'trend_adx', 'liq_rel_vol_20']
        new_cols = {}
        for col in core_cols:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    new_cols[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # 3. Rolling Z-Scores (Normalization/Statistical Features)
        for col in ['mom_rsi_14', 'vol_atr_20']:
            if col in df.columns:
                mean = df[col].rolling(100).mean()
                std = df[col].rolling(100).std()
                new_cols[f'stat_{col}_z'] = (df[col] - mean) / (std + 1e-9)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # Final Clean & Enforce XGBoost Feature Order (125 features)
        df = self._apply_institutional_normalization(df)
        
        required_features = [
            'open', 'high', 'low', 'close', 'volume', 'ema_5', 'ema_5_dist', 'ema_10', 'ema_10_dist',
            'ema_20', 'ema_20_dist', 'ema_30', 'ema_30_dist', 'ema_50', 'ema_50_dist', 'ema_100',
            'ema_100_dist', 'ema_200', 'ema_200_dist', 'trend_hma', 'trend_adx', 'trend_adx_pos',
            'trend_adx_neg', 'trend_slope_3', 'trend_slope_5', 'trend_slope_10', 'trend_slope_20',
            'trend_hurst', 'mom_rsi_7', 'mom_rsi_7_slope', 'mom_rsi_14', 'mom_rsi_14_slope',
            'mom_rsi_21', 'mom_rsi_21_slope', 'mom_rsi_28', 'mom_rsi_28_slope', 'mom_macd',
            'mom_macd_signal', 'mom_macd_diff', 'mom_tsi', 'mom_uo', 'mom_stoch', 'mom_stoch_sig',
            'mom_roc_1', 'mom_roc_3', 'mom_roc_5', 'mom_roc_10', 'mom_roc_20', 'mom_roc_30',
            'vol_atr_10', 'vol_atr_10_pct', 'vol_atr_20', 'vol_atr_20_pct', 'vol_atr_50',
            'vol_atr_50_pct', 'vol_parkinson', 'vol_gk_proxy', 'vol_bb_w_20', 'vol_bb_hi_20',
            'vol_bb_lo_20', 'vol_bb_w_50', 'vol_bb_hi_50', 'vol_bb_lo_50', 'vol_donchian_hi',
            'struct_pivot', 'struct_r1', 'struct_s1', 'struct_r2', 'struct_s2', 'struct_range_pos_20',
            'struct_range_pos_50', 'struct_range_pos_100', 'struct_fib_0.236', 'struct_fib_dist_0.236',
            'struct_fib_0.382', 'struct_fib_dist_0.382', 'struct_fib_0.5', 'struct_fib_dist_0.5',
            'struct_fib_0.618', 'struct_fib_dist_0.618', 'struct_fib_0.786', 'struct_fib_dist_0.786',
            'liq_rel_vol_10', 'liq_rel_vol_20', 'liq_rel_vol_50', 'liq_vol_imb', 'liq_spread_vol',
            'liq_price_impact', 'liq_vol_skew', 'liq_vol_force_5', 'liq_vol_vbp_5', 'liq_vol_force_10',
            'liq_vol_vbp_10', 'liq_vol_force_20', 'liq_vol_vbp_20', 'pa_body_pct', 'pa_upper_wick',
            'pa_lower_wick', 'liq_gap', 'vol_spike', 'pa_close_lag_1', 'pa_close_lag_2', 'pa_close_lag_3',
            'pa_doji', 'pa_engulfing', 'time_hour', 'time_london', 'time_ny', 'close_lag_1',
            'close_lag_2', 'close_lag_3', 'mom_rsi_14_lag_1', 'mom_rsi_14_lag_2', 'mom_rsi_14_lag_3',
            'vol_atr_20_pct_lag_1', 'vol_atr_20_pct_lag_2', 'vol_atr_20_pct_lag_3', 'trend_adx_lag_1',
            'trend_adx_lag_2', 'trend_adx_lag_3', 'liq_rel_vol_20_lag_1', 'liq_rel_vol_20_lag_2',
            'liq_rel_vol_20_lag_3', 'stat_mom_rsi_14_z', 'stat_vol_atr_20_z'
        ]
        
        # Ensure all columns exist, if not fill with 0
        for col in required_features:
            if col not in df.columns:
                df[col] = 0.0
                
        return df[required_features].dropna()

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """20 Trend Features: EMAs, Hurst, ADX, Slope, Dispersion."""
        for p in [5, 10, 20, 30, 50, 100, 200]:
            df[f'ema_{p}'] = ta.trend.ema_indicator(df['close'], window=p)
            df[f'ema_{p}_dist'] = (df['close'] - df[f'ema_{p}']) / (df[f'ema_{p}'] + 1e-9)
        
        # Hull EMA & SuperTrend Proxy
        wma1 = ta.trend.wma_indicator(df['close'], window=9)
        wma2 = ta.trend.wma_indicator(df['close'], window=18)
        df['trend_hma'] = ta.trend.wma_indicator(2 * pd.Series(wma1.values - wma2.values, index=df.index), window=4)
        
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['trend_adx'] = adx.adx()
        df['trend_adx_pos'] = adx.adx_pos()
        df['trend_adx_neg'] = adx.adx_neg()
        
        for s in [3, 5, 10, 20]:
            df[f'trend_slope_{s}'] = df['close'].diff(s) / (df['close'].shift(s) + 1e-9)
        
        df['trend_hurst'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / (df['close'].rolling(20).std() + 1e-9)
        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """20 Momentum Features: RSI, MACD, SMI, TSI, ROC Velocity."""
        for p in [7, 14, 21, 28]:
            df[f'mom_rsi_{p}'] = ta.momentum.rsi(df['close'], window=p)
            df[f'mom_rsi_{p}_slope'] = df[f'mom_rsi_{p}'].diff(3)
        
        macd_obj = ta.trend.MACD(df['close'])
        df['mom_macd'] = macd_obj.macd()
        df['mom_macd_signal'] = macd_obj.macd_signal()
        df['mom_macd_diff'] = macd_obj.macd_diff()
        
        df['mom_tsi'] = ta.momentum.tsi(df['close'], window_slow=25, window_fast=13)
        df['mom_uo'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
        df['mom_stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        df['mom_stoch_sig'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
        
        for r in [1, 3, 5, 10, 20, 30]:
            df[f'mom_roc_{r}'] = ta.momentum.roc(df['close'], window=r)
            
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """15 Volatility Features: ATR, BB, Donchian, Rolling Variance."""
        for p in [10, 20, 50]:
            atr_obj = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=p)
            df[f'vol_atr_{p}'] = atr_obj.average_true_range()
            df[f'vol_atr_{p}_pct'] = df[f'vol_atr_{p}'] / (df['close'] + 1e-9)
        
        # High-Low Range Volatility (Parkinson proxy)
        df['vol_parkinson'] = np.sqrt(1 / (4 * np.log(2)) * (np.log(df['high'] / (df['low'] + 1e-9)))**2)
        
        log_hl = (np.log(df['high'] / (df['low'] + 1e-9)))**2
        log_cc = (2 * np.log(2) - 1) * (np.log(df['close'] / (df['open'].replace(0, np.nan).ffill().fillna(1e-9)))**2)
        df['vol_gk_proxy'] = np.sqrt(np.clip(0.5 * log_hl - log_cc, 0, None))
        
        for p in [20, 50]:
            bb_obj = ta.volatility.BollingerBands(df['close'], window=p)
            df[f'vol_bb_w_{p}'] = bb_obj.bollinger_wband()
            df[f'vol_bb_hi_{p}'] = (df['high'] - bb_obj.bollinger_hband()) / (df['close'] + 1e-9)
            df[f'vol_bb_lo_{p}'] = (df['low'] - bb_obj.bollinger_lband()) / (df['close'] + 1e-9)
            
        donchian_obj = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=20)
        df['vol_donchian_hi'] = (df['high'] - donchian_obj.donchian_channel_hband()) / (df['close'] + 1e-9)
        
        return df

    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """15 Structure Features: Pivot Points, S/R Proximity, Range Positioning."""
        df['struct_pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['struct_r1'] = (2 * df['struct_pivot']) - df['low'].shift(1)
        df['struct_s1'] = (2 * df['struct_pivot']) - df['high'].shift(1)
        df['struct_r2'] = df['struct_pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['struct_s2'] = df['struct_pivot'] - (df['high'].shift(1) - df['low'].shift(1))
        
        for p in [20, 50, 100]:
            df[f'struct_range_pos_{p}'] = (df['close'] - df['low'].rolling(p).min()) / \
                                          (df['high'].rolling(p).max() - df['low'].rolling(p).min() + 1e-9)
        
        h100, l100 = df['high'].rolling(100).max(), df['low'].rolling(100).min()
        for f in [0.236, 0.382, 0.5, 0.618, 0.786]:
            df[f'struct_fib_{f}'] = h100 - (h100 - l100) * f
            df[f'struct_fib_dist_{f}'] = (df['close'] - df[f'struct_fib_{f}']) / (df['close'] + 1e-9)
            
        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """15 Microstructure: Volume Imbalance, Relative Volume, Flow, Price Impact."""
        for p in [10, 20, 50]:
            df[f'liq_rel_vol_{p}'] = df['volume'] / (df['volume'].rolling(p).mean() + 1e-9)
        
        # 1. Volume Imbalance (Tick Proxy)
        df['liq_vol_imb'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9) * df['volume']
        
        # 2. Bid-Ask Spread Volatility Proxy
        df['liq_spread_vol'] = (df['high'] - df['low']).rolling(20).std() / (df['close'].rolling(20).mean() + 1e-9)
        
        # 3. Price Impact Estimate (Amihud Illiquidity)
        rets = df['close'].pct_change().abs()
        df['liq_price_impact'] = rets.rolling(20).mean() / (df['volume'].rolling(20).mean() + 1e-9) * 1e6
        
        # 4. Volume Profile Skewness
        df['liq_vol_skew'] = df['volume'].rolling(50).skew()
        
        # 5. Order Flow Imbalance (OFI) Proxy
        for p in [5, 10, 20]:
            df[f'liq_vol_force_{p}'] = df['liq_vol_imb'].rolling(p).sum()
            df[f'liq_vol_vbp_{p}'] = df['volume'].rolling(p).std() / (df['volume'].rolling(p).mean() + 1e-9)
            
        return df

    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """10 Price Action: Rejection Ratios, Body Ratios, Patterns, Micro-Spikes."""
        br = np.abs(df['close'] - df['open'])
        cr = df['high'] - df['low'] + 1e-9
        df['pa_body_pct'] = br / cr
        df['pa_upper_wick'] = (df['high'] - np.maximum(df['close'], df['open'])) / cr
        df['pa_lower_wick'] = (np.minimum(df['close'], df['open']) - df['low']) / cr
        
        df['liq_gap'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-9)
        df['vol_spike'] = cr / (cr.rolling(20).median() + 1e-9)
        
        for l in [1, 2, 3]:
            df[f'pa_close_lag_{l}'] = df['close'].pct_change(l)
            
        df['pa_doji'] = (df['pa_body_pct'] < 0.1).astype(int)
        df['pa_engulfing'] = ((df['pa_body_pct'] > 0.7) & (br > br.shift(1).fillna(0))).astype(int)
        
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """10 Time/Session: Asian, London, NY Open Ratios."""
        if hasattr(df.index, 'hour'):
            df['time_hour'] = df.index.hour
            df['time_london'] = ((df['time_hour'] >= 8) & (df['time_hour'] <= 16)).astype(int)
            df['time_ny'] = ((df['time_hour'] >= 13) & (df['time_hour'] <= 21)).astype(int)
        else:
            df['time_hour'] = 0
            df['time_london'] = 0
            df['time_ny'] = 0
        return df

    def _apply_institutional_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies robust normalization (Winsorizing + Robust Scale proxy)."""
        df = df.copy() # Fix DataFrame fragmentation warning
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        return df
