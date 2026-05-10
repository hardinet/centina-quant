from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CryptoEvent:
    name:            str
    date:            str
    category:        str       # halving / upgrade / crash / regulatory / etf
    impact_score:    float     # 0-10
    affected_symbols: list[str]
    description:     str = ""


KNOWN_EVENTS: list[CryptoEvent] = [
    CryptoEvent("Bitcoin Halving 1",         "2012-11-28", "halving",    9, ["BTCUSDT"]),
    CryptoEvent("Bitcoin Halving 2",         "2016-07-09", "halving",    9, ["BTCUSDT"]),
    CryptoEvent("Bitcoin Halving 3",         "2020-05-11", "halving",    9, ["BTCUSDT"]),
    CryptoEvent("Bitcoin Halving 4",         "2024-04-19", "halving",    9, ["BTCUSDT"]),
    CryptoEvent("Ethereum The Merge",        "2022-09-15", "upgrade",    8, ["ETHUSDT"]),
    CryptoEvent("MtGox Collapse",            "2014-02-07", "crash",     10, ["BTCUSDT"]),
    CryptoEvent("DAO Hack",                  "2016-06-17", "crash",      7, ["ETHUSDT"]),
    CryptoEvent("COVID Crash",               "2020-03-12", "crash",     10, ["BTCUSDT","ETHUSDT"]),
    CryptoEvent("Terra/Luna Collapse",       "2022-05-09", "crash",     10, ["BTCUSDT","ETHUSDT"]),
    CryptoEvent("FTX Collapse",              "2022-11-08", "crash",     10, ["BTCUSDT","ETHUSDT"]),
    CryptoEvent("Bitcoin ETF Approval",      "2024-01-10", "etf",        9, ["BTCUSDT"]),
    CryptoEvent("China Ban (2017)",          "2017-09-04", "regulatory", 8, ["BTCUSDT"]),
    CryptoEvent("China Ban (2021)",          "2021-09-24", "regulatory", 8, ["BTCUSDT"]),
    CryptoEvent("Russia-Ukraine War",        "2022-02-24", "macro",      6, ["BTCUSDT"]),
    CryptoEvent("Fed Rate Hike Campaign",    "2022-03-16", "macro",      7, ["BTCUSDT","ETHUSDT"]),
    CryptoEvent("Bitcoin ATH Dec 2020",      "2020-12-16", "ath",        7, ["BTCUSDT"]),
    CryptoEvent("Bitcoin ATH Nov 2021",      "2021-11-10", "ath",        8, ["BTCUSDT"]),
    CryptoEvent("DeFi Summer 2020",          "2020-07-01", "trend",      7, ["ETHUSDT"]),
    CryptoEvent("Alt Season Q1 2021",        "2021-01-01", "alt_season", 8, ["ETHUSDT"]),
    CryptoEvent("NFT Boom 2021",             "2021-03-01", "trend",      6, ["ETHUSDT"]),
]


class EventDataFetcher:

    def get_events(
        self,
        start_date: str,
        end_date: str,
        category: str | None = None,
    ) -> list[CryptoEvent]:
        events = [e for e in KNOWN_EVENTS
                  if start_date <= e.date <= end_date]
        if category:
            events = [e for e in events if e.category == category]
        return sorted(events, key=lambda e: e.date)

    def get_halving_cycles(self) -> list[dict]:
        halvings = [e for e in KNOWN_EVENTS if e.category == "halving"]
        cycles = []
        for i, h in enumerate(halvings):
            next_h = halvings[i+1].date if i + 1 < len(halvings) else "2028-04-01"
            cycles.append({
                "halving_date":  h.date,
                "next_halving":  next_h,
                "cycle_number":  i + 1,
            })
        return cycles

    def get_crash_events(self) -> list[CryptoEvent]:
        return [e for e in KNOWN_EVENTS if e.category == "crash"]

    def as_dataframe(self, events: list[CryptoEvent]) -> pd.DataFrame:
        return pd.DataFrame([{
            "date":    e.date,
            "name":    e.name,
            "category": e.category,
            "impact":  e.impact_score,
        } for e in events])
