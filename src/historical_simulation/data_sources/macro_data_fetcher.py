from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

FRED_SERIES = {
    "DFF":                "Federal Funds Rate",
    "CPIAUCSL":           "CPI (Inflation)",
    "UNRATE":             "Unemployment Rate",
    "T10Y2Y":             "10Y-2Y Treasury Spread",
    "DGS10":              "10Y Treasury Rate",
    "VIXCLS":             "VIX Index",
    "DTWEXBGS":           "US Dollar Index",
    "GOLDPMGBD228NLBM":   "Gold Price",
}


class MacroDataFetcher:
    """
    Fetches macroeconomic variables from FRED API.
    Aligns to daily crypto data with forward-fill.
    """

    def __init__(self, api_key: str = ""):
        self._key = api_key

    def get_macro_series(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        try:
            import requests
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id":        series_id,
                "observation_start": start_date,
                "observation_end":   end_date,
                "api_key":          self._key or "demo",
                "file_type":        "json",
            }
            r = requests.get(url, params=params, timeout=30)
            data = r.json().get("observations", [])
            if not data:
                return pd.Series(dtype=float, name=series_id)
            df = pd.DataFrame(data)[["date", "value"]]
            df["date"]  = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.set_index("date")["value"].rename(series_id)
        except Exception as e:
            logger.warning("FRED fetch failed for %s: %s", series_id, e)
            return pd.Series(dtype=float, name=series_id)

    def get_all_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        series = {}
        for sid in FRED_SERIES:
            s = self.get_macro_series(sid, start_date, end_date)
            if not s.empty:
                series[sid] = s
        if not series:
            return pd.DataFrame()
        df = pd.DataFrame(series)
        # Daily reindex, forward fill
        date_idx = pd.date_range(start=start_date, end=end_date, freq="D")
        df = df.reindex(date_idx).ffill().bfill()
        return df

    def correlate_with_crypto(
        self,
        crypto_series: pd.Series,
        macro_var: str,
        start_date: str,
        end_date: str,
        lag_days: int = 0,
    ) -> float:
        macro = self.get_macro_series(macro_var, start_date, end_date)
        if macro.empty:
            return float("nan")
        aligned = pd.concat([crypto_series, macro], axis=1).dropna()
        if len(aligned) < 10:
            return float("nan")
        if lag_days:
            aligned.iloc[:, 1] = aligned.iloc[:, 1].shift(lag_days)
            aligned = aligned.dropna()
        return float(aligned.corr().iloc[0, 1])
