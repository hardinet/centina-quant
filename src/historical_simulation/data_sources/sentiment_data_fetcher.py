from __future__ import annotations

import logging
from datetime import datetime, date

import pandas as pd

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/"


class SentimentDataFetcher:

    def get_fear_greed(self, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import requests
            from datetime import datetime as dt

            start_ts = int(dt.fromisoformat(start_date).timestamp())
            end_ts   = int(dt.fromisoformat(end_date).timestamp())
            days     = (dt.fromisoformat(end_date) - dt.fromisoformat(start_date)).days + 1

            r = requests.get(FEAR_GREED_URL, params={"limit": days, "format": "json"}, timeout=30)
            data = r.json().get("data", [])
            records = []
            for item in data:
                ts = int(item.get("timestamp", 0))
                if start_ts <= ts <= end_ts:
                    records.append({
                        "date":  pd.to_datetime(ts, unit="s", utc=True),
                        "fear_greed": int(item.get("value", 50)),
                        "classification": item.get("value_classification", ""),
                    })
            if not records:
                return pd.DataFrame()
            df = pd.DataFrame(records).set_index("date").sort_index()
            return df
        except Exception as e:
            logger.debug("Fear & Greed fetch failed: %s", e)
            return pd.DataFrame()

    def get_google_trends(self, keywords: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl="en-US", tz=0)
            pytrends.build_payload(keywords[:5], cat=0, timeframe=f"{start_date} {end_date}")
            df = pytrends.interest_over_time()
            return df.drop(columns=["isPartial"], errors="ignore") if not df.empty else pd.DataFrame()
        except Exception as e:
            logger.debug("Google Trends fetch failed: %s", e)
            return pd.DataFrame()

    def get_funding_rates(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            from binance import Client
            from datetime import datetime as dt
            client = Client("", "")
            start_ms = int(dt.fromisoformat(start_date).timestamp() * 1000)
            end_ms   = int(dt.fromisoformat(end_date).timestamp() * 1000)
            rates = client.get_funding_rate(
                symbol=symbol, startTime=start_ms, endTime=end_ms, limit=1000
            )
            if not rates:
                return pd.DataFrame()
            df = pd.DataFrame(rates)
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            df["fundingRate"] = df["fundingRate"].astype(float)
            return df.set_index("fundingTime")[["fundingRate"]]
        except Exception as e:
            logger.debug("Funding rates fetch failed: %s", e)
            return pd.DataFrame()
