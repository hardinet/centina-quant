"""Unit tests for SyntheticDataGenerator and GapFiller."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.historical_simulation.synthetic.synthetic_data_generator import (
    SyntheticConfig,
    SyntheticDataGenerator,
)
from src.historical_simulation.synthetic.gap_filler import GapFiller


@pytest.fixture
def gen() -> SyntheticDataGenerator:
    return SyntheticDataGenerator()


class TestSyntheticDataGenerator:

    def test_gbm_returns_dataframe(self, gen):
        cfg = SyntheticConfig(model="gbm", n_days=30, seed=1)
        df = gen.generate(cfg)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_heston_model(self, gen):
        cfg = SyntheticConfig(model="heston", n_days=30, seed=2)
        df = gen.generate(cfg)
        assert "close" in df.columns
        assert (df["close"] > 0).all()

    def test_garch_model(self, gen):
        cfg = SyntheticConfig(model="garch", n_days=30, seed=3)
        df = gen.generate(cfg)
        assert "close" in df.columns

    def test_bootstrap_model(self, gen):
        cfg = SyntheticConfig(model="bootstrap", n_days=30, seed=4)
        df = gen.generate(cfg)
        assert not df.empty

    def test_ohlcv_columns(self, gen):
        df = gen.generate()
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_high_gte_close(self, gen):
        df = gen.generate(SyntheticConfig(seed=5))
        assert (df["high"] >= df["close"]).all()

    def test_low_lte_close(self, gen):
        df = gen.generate(SyntheticConfig(seed=6))
        assert (df["low"] <= df["close"]).all()

    def test_volume_positive(self, gen):
        df = gen.generate(SyntheticConfig(seed=7))
        assert (df["volume"] > 0).all()

    def test_n_days_respected_hourly(self, gen):
        cfg = SyntheticConfig(n_days=10, freq="1h", seed=8)
        df = gen.generate(cfg)
        assert len(df) == 10 * 24

    def test_start_price_approx(self, gen):
        cfg = SyntheticConfig(start_price=50_000, n_days=5, seed=9)
        df = gen.generate(cfg)
        assert abs(df["close"].iloc[0] - 50_000) / 50_000 < 0.05

    def test_reproducible_with_seed(self, gen):
        cfg = SyntheticConfig(seed=42, n_days=10)
        df1 = gen.generate(cfg)
        df2 = gen.generate(cfg)
        pd.testing.assert_frame_equal(df1, df2)

    def test_multi_asset_shape(self, gen):
        dfs = gen.generate_multi_asset(n_assets=3)
        assert len(dfs) == 3
        for df in dfs:
            assert "close" in df.columns

    def test_crash_scenario(self, gen):
        cfg = SyntheticConfig(n_days=200, seed=10)
        df = gen.generate_crash_scenario(crash_pct=-0.40, crash_day=60, cfg=cfg)
        assert not df.empty
        assert "close" in df.columns


class TestGapFiller:

    def _make_gapped_df(self) -> pd.DataFrame:
        gen = SyntheticDataGenerator()
        cfg = SyntheticConfig(n_days=30, seed=20)
        df = gen.generate(cfg)
        # Introduce gap
        df.iloc[100:115] = np.nan
        return df

    def test_fill_removes_nans(self):
        filler = GapFiller()
        df = self._make_gapped_df()
        filled = filler.fill(df, method="gbm", max_gap_hours=48)
        assert filled["close"].isna().sum() == 0

    def test_validate_returns_dict(self):
        filler = GapFiller()
        gen = SyntheticDataGenerator()
        df = gen.generate(SyntheticConfig(n_days=10, seed=21))
        result = filler.validate(df)
        assert "total_rows" in result
        assert "completeness" in result
        assert 0 <= result["completeness"] <= 1

    def test_fill_volume_gaps(self):
        filler = GapFiller()
        gen = SyntheticDataGenerator()
        df = gen.generate(SyntheticConfig(n_days=10, seed=22))
        df.iloc[10:15, df.columns.get_loc("volume")] = 0
        filled = filler.fill_volume_gaps(df)
        assert (filled["volume"] > 0).all()
