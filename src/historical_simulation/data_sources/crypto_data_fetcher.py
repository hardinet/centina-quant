from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DB = Path("data/historical_cache.db")


class CryptoDataFetcher:
    """
    Multi-source OHLCV fetcher with SQLite cache.
    Sources: Binance (2017+), CoinGecko (2013+), Yahoo Finance.
    """

    SUPPORTED_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]

    def __init__(self):
        self._init_cache()

    def get_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        df = self._fetch_from_best_source(symbol, start_date, end_date, timeframe)
        if df is not None and not df.empty:
            df = self._fill_gaps(df)
            df = df.tz_localize(None) if df.index.tz is not None else df
            self._write_cache(cache_key, df)
        return df or pd.DataFrame()

    def get_available_symbols(self) -> list[str]:
        try:
            from binance import Client
            client = Client("", "")
            info = client.get_exchange_info()
            return [s["symbol"] for s in info["symbols"]
                    if s["symbol"].endswith("USDT") and s["status"] == "TRADING"]
        except Exception:
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]

    def get_date_range(self, symbol: str) -> tuple[str, str]:
        """Returns (earliest_date, latest_date) available for symbol."""
        if "BTC" in symbol:
            return "2017-08-17", date.today().isoformat()
        if "ETH" in symbol:
            return "2017-08-17", date.today().isoformat()
        return "2019-01-01", date.today().isoformat()

    def _fetch_from_best_source(self, symbol: str, start: str, end: str, tf: str) -> pd.DataFrame | None:
        # 1. Try Binance
        df = self._from_binance(symbol, start, end, tf)
        if df is not None and not df.empty:
            return df

        # 2. Try yfinance for BTC/ETH daily
        if tf in ("1d", "1w") and symbol in ("BTCUSDT", "ETHUSDT"):
            df = self._from_yfinance(symbol, start, end, tf)
            if df is not None and not df.empty:
                return df

        # 3. Try CoinGecko daily
        if tf == "1d":
            df = self._from_coingecko(symbol, start, end)
            if df is not None and not df.empty:
                return df

        return None

    def _from_binance(self, symbol: str, start: str, end: str, tf: str) -> pd.DataFrame | None:
        try:
            from binance import Client
            client = Client("", "")
            klines = client.get_historical_klines(
                symbol, tf,
                start_str=start,
                end_str=end,
            )
            if not klines:
                return None
            cols = ["open_time","open","high","low","close","volume",
                    "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"]
            df = pd.DataFrame(klines, columns=cols)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for c in ("open","high","low","close","volume"):
                df[c] = df[c].astype(float)
            df = df.set_index("open_time")[["open","high","low","close","volume"]]
            return df
        except Exception as e:
            logger.debug("Binance fetch failed for %s: %s", symbol, e)
            return None

    def _from_yfinance(self, symbol: str, start: str, end: str, tf: str) -> pd.DataFrame | None:
        try:
            import yfinance as yf
            ticker = symbol.replace("USDT", "-USD")
            interval = "1d" if tf in ("1d", "1w") else tf
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index, utc=True)
            return df[["open","high","low","close","volume"]]
        except Exception as e:
            logger.debug("yfinance fetch failed: %s", e)
            return None

    def _from_coingecko(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        try:
            import requests
            coin_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum",
                        "BNBUSDT": "binancecoin", "SOLUSDT": "solana"}
            coin_id = coin_map.get(symbol)
            if not coin_id:
                return None

            start_ts = int(datetime.fromisoformat(start).timestamp())
            end_ts   = int(datetime.fromisoformat(end).timestamp())
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
            r = requests.get(url, params={"vs_currency":"usd","from":start_ts,"to":end_ts}, timeout=30)
            data = r.json()

            prices = pd.DataFrame(data["prices"], columns=["ts","close"])
            vols   = pd.DataFrame(data.get("total_volumes",[]), columns=["ts","volume"])
            prices["ts"] = pd.to_datetime(prices["ts"], unit="ms", utc=True)
            vols["ts"]   = pd.to_datetime(vols["ts"], unit="ms", utc=True)
            df = prices.set_index("ts").join(vols.set_index("ts"))
            df["open"] = df["close"].shift(1)
            df["high"] = df["close"]
            df["low"]  = df["close"]
            return df[["open","high","low","close","volume"]].dropna()
        except Exception as e:
            logger.debug("CoinGecko fetch failed: %s", e)
            return None

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.sort_index()
        freq_map = {"1m":"1min","5m":"5min","15m":"15min","1h":"1h","4h":"4h","1d":"1d","1w":"1W"}
        # Forward fill for small gaps
        df = df.ffill(limit=3)
        # Linear interpolation for remaining NaN
        return df.interpolate(method="linear", limit=10)

    def _read_cache(self, key: str) -> pd.DataFrame | None:
        try:
            conn = sqlite3.connect(CACHE_DB)
            rows = conn.execute(
                "SELECT data_json FROM ohlcv_cache WHERE cache_key=?", (key,)
            ).fetchone()
            conn.close()
            if rows:
                return pd.read_json(rows[0])
        except Exception:
            pass
        return None

    def _write_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            conn = sqlite3.connect(CACHE_DB)
            conn.execute(
                "INSERT OR REPLACE INTO ohlcv_cache (cache_key, data_json, cached_at) VALUES (?,?,?)",
                (key, df.to_json(), datetime.utcnow().isoformat()),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _init_cache(self) -> None:
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(CACHE_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_cache (
                cache_key TEXT PRIMARY KEY, data_json TEXT, cached_at TEXT
            )
        """)
        conn.commit()
        conn.close()
