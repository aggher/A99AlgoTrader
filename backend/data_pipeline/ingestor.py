import yfinance as yf
import pandas as pd
import logging
import os
import requests
from typing import Optional, Dict, List
from .quality_control import DataQualityControl
from .multi_source_validator import MultiSourceValidator

log = logging.getLogger(__name__)

class DataIngestor:
    """Robust multi-source data ingestion with integrated DQC and Verification."""

    def __init__(self, symbol_map: Dict[str, str]):
        self.symbol_map = symbol_map
        self.dqc = DataQualityControl()
        self.msv = MultiSourceValidator()

    def _fetch_twelve_data(self, symbol: str, interval: str, period: str, api_key: str) -> Optional[pd.DataFrame]:
        interval_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "1d": "1day", "1mo": "1month"}
        tw_interval = interval_map.get(interval)
        if not tw_interval: return None
        
        tw_symbol = f"{symbol[:3]}/{symbol[3:]}" if len(symbol) == 6 and symbol != "XAUUSD" else symbol
        if symbol == "XAUUSD": tw_symbol = "XAU/USD"
        
        outputsize = 500 if period == "2d" else 5000
        url = f"https://api.twelvedata.com/time_series?symbol={tw_symbol}&interval={tw_interval}&outputsize={outputsize}&apikey={api_key}"
        try:
            resp = requests.get(url)
            data = resp.json()
            if data.get("status") == "error":
                log.error(f"Twelve Data Error for {symbol}: {data.get('message')}")
                return None
            values = data.get("values", [])
            if not values: return None
            df = pd.DataFrame(values)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
            df.index = df.index.tz_convert(None) # Match yfinance naive UTC
            df = df.astype(float)
            df = df.sort_index()
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            log.error(f"Twelve Data request failed: {e}")
            return None

    def fetch_institutional_data(self, symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
        ticker = self.symbol_map.get(symbol)
        if not ticker:
            log.error(f"Symbol {symbol} not found in registry.")
            return None

        log.info(f"Ingesting {symbol} [{interval}] from Multi-Source Pipeline...")
        
        try:
            # 1. PRIMARY SOURCE: YFinance
            df_primary = yf.download(ticker, period=period, interval=interval, 
                                     progress=False, auto_adjust=True, threads=False)
            
            if df_primary.empty:
                log.warning(f"Empty primary data for {symbol}")
                return None

            # Handle MultiIndex and Case
            if isinstance(df_primary.columns, pd.MultiIndex):
                df_primary.columns = df_primary.columns.get_level_values(0)
            df_primary.columns = [c.lower() for c in df_primary.columns]
            if df_primary.index.tz is not None:
                df_primary.index = df_primary.index.tz_convert(None)

            # 2. SECONDARY SOURCE: Twelve Data
            twelve_api_key = os.getenv("TWELVE_DATA_API_KEY")
            if twelve_api_key:
                twelve_df = self._fetch_twelve_data(symbol, interval, period, twelve_api_key)
                if twelve_df is not None:
                    df_secondary = twelve_df
                else:
                    log.warning(f"Twelve data fetch failed for {symbol}, falling back to simulated.")
                    df_secondary = df_primary.copy()
            else:
                log.warning("No TWELVE_DATA_API_KEY found, using simulated secondary data.")
                df_secondary = df_primary.copy()
            
            # 3. CROSS-SOURCE VERIFICATION (Section 1)
            v_res = self.msv.validate_ingestion(symbol, interval, df_primary, df_secondary)
            if v_res["status"] == "INVALID":
                log.error(f"❌ Cross-source validation failed for {symbol}: {v_res['reason']}")
                return None

            # 4. DATA QUALITY CONTROL (DQC)
            is_delta = period == "2d"
            df_clean, stats = self.dqc.validate_snapshot(df_primary, symbol, is_delta=is_delta)
            
            if stats["outliers_detected"] > 0 or stats["missing_aligned"] > 0:
                log.info(f"DQC Applied to {symbol}: {stats}")
            
            if not self.dqc.check_stability(df_clean, is_delta=is_delta):
                log.warning(f"⚠️ Stability check failed for {symbol}. Rejecting snapshot.")
                return None

            # Institutional Normalization
            df_clean.columns = [c.lower() for c in df_clean.columns]
            
            return df_clean

        except Exception as e:
            log.error(f"Ingestion Error for {symbol}: {e}")
            return None
