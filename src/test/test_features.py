"""Tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forex.config import FeatureConfig
from forex.features.engineering import build_features, _rsi, _bb_pct

from tests.conftest import make_forex_df


class TestBuildFeatures:
    def test_returns_dataframe(self):
        df = make_forex_df(n_rows=300)
        cfg = FeatureConfig()
        result = build_features(df, cfg)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_in_output(self):
        df = make_forex_df(n_rows=300)
        cfg = FeatureConfig()
        result = build_features(df, cfg)
        assert result.isna().sum().sum() == 0, "build_features must drop NaN warm-up rows"

    def test_no_raw_ohlc_in_output(self):
        df = make_forex_df(n_rows=300)
        cfg = FeatureConfig()
        result = build_features(df, cfg)
        assert "EURPLN_CLOSE" not in result.columns
        assert "timestamp" not in result.columns

    def test_time_features_present(self):
        df = make_forex_df(n_rows=300)
        cfg = FeatureConfig()
        result = build_features(df, cfg)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert col in result.columns

    def test_feature_count_is_deterministic(self):
        df = make_forex_df(n_rows=300)
        cfg = FeatureConfig()
        result1 = build_features(df, cfg)
        result2 = build_features(df, cfg)
        assert result1.shape[1] == result2.shape[1]

    def test_skips_missing_aux_pair_gracefully(self):
        df = make_forex_df(n_rows=300)
        df = df.drop(columns=[c for c in df.columns if c.startswith("PLNUSD")])
        cfg = FeatureConfig()
        # Should not raise
        result = build_features(df, cfg)
        assert isinstance(result, pd.DataFrame)

    def test_hour_features_bounded(self):
        df = make_forex_df(n_rows=300)
        cfg = FeatureConfig()
        result = build_features(df, cfg)
        assert result["hour_sin"].between(-1.0, 1.0).all()
        assert result["hour_cos"].between(-1.0, 1.0).all()


class TestIndicators:
    def test_rsi_bounded(self):
        prices = pd.Series(np.cumsum(np.random.normal(0, 0.01, 200)) + 4.0)
        rsi = _rsi(prices, period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_bb_pct_near_center_for_stable_price(self):
        prices = pd.Series([4.0] * 100)
        pct = _bb_pct(prices, period=20, n_std=2.0)
        # Constant price → std≈0, numerator≈0 → %B ≈ 0 (clipped by epsilon)
        assert pct.dropna().abs().max() < 1.0
