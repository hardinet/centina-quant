from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"

ONCHAIN_METRICS = {
    "active_addresses":   ("addresses", "count"),
    "transaction_volume": ("transactions", "transfers_volume_sum"),
    "hash_rate":          ("mining", "hash_rate_mean"),
    "exchange_inflow":    ("transactions", "transfers_volume_exchanges_net"),
    "mvrv_ratio":         ("indicators", "mvrv"),
    "nvt_ratio":          ("indicators", "nvt"),
    "sopr":               ("indicators", "sopr"),
}


class OnchainDataFetcher:
    """
    Fetches on-chain metrics from Glassnode (requires API key).
    Falls back to empty DataFrame if no key / API unreachable.
    """

    def __init__(self, api_key: str = ""):
        self._key = api_key or os.getenv("GLASSNODE_API_KEY", "")

    def get_onchain_metric(
        self, metric_name: str, symbol: str, start_date: str, end_date: str
    ) -> pd.Series:
        if not self._key:
            logger.debug("OnchainDataFetcher: no Glassnode API key")
            return pd.Series(dtype=float, name=metric_name)

        if metric_name not in ONCHAIN_METRICS:
            logger.warning("Unknown metric: %s", metric_name)
            return pd.Series(dtype=float, name=metric_name)

        try:
            import requests
            from datetime import datetime
            category, endpoint = ONCHAIN_METRICS[metric_name]
            url = f"{GLASSNODE_BASE}/{category}/{endpoint}"
            params = {
                "a":     symbol.replace("USDT",""),
                "s":     int(datetime.fromisoformat(start_date).timestamp()),
                "u":     int(datetime.fromisoformat(end_date).timestamp()),
                "i":     "24h",
                "api_key": self._key,
            }
            r = requests.get(url, params=params, timeout=30)
            data = r.json()
            if not isinstance(data, list):
                return pd.Series(dtype=float, name=metric_name)
            df = pd.DataFrame(data)
            df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
            return df.set_index("t")["v"].rename(metric_name)
        except Exception as e:
            logger.debug("Glassnode fetch failed for %s: %s", metric_name, e)
            return pd.Series(dtype=float, name=metric_name)

    def get_all_onchain(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        series = {}
        for name in ONCHAIN_METRICS:
            s = self.get_onchain_metric(name, symbol, start_date, end_date)
            if not s.empty:
                series[name] = s
        return pd.DataFrame(series) if series else pd.DataFrame()
