"""Tests for data loading and preprocessing."""
import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_forex_df

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forex.config import DataConfig
from forex.data.loader import drop_sparse_columns
from forex.data.preprocessing import create_target, handle_missing, split_time_aware


class TestDropSparseColumns:
    def test_drops_below_threshold(self):
        df = make_forex_df(n_rows=200, sparse_pairs=["PLNHUF"])
        n_before = df.shape[1]
        result = drop_sparse_columns(df, min_coverage=0.30)
        # PLNHUF_* columns should be dropped (80% NaN = 20% coverage)
        assert all("PLNHUF" not in c for c in result.columns)
        assert result.shape[1] < n_before

    def test_always_keeps_eurpln(self):
        df = make_forex_df(n_rows=200)
        result = drop_sparse_columns(df, min_coverage=0.99)
        eurpln_cols = [c for c in result.columns if c.startswith("EURPLN")]
        assert len(eurpln_cols) == 4, "EURPLN OHLC must always be preserved"

    def test_keeps_timestamp(self):
        df = make_forex_df(n_rows=200)
        result = drop_sparse_columns(df, min_coverage=0.30)
        assert "timestamp" in result.columns


class TestHandleMissing:
    def test_drops_rows_with_nan_eurpln_close(self):
        df = make_forex_df(n_rows=100)
        df.loc[5:10, "EURPLN_CLOSE"] = np.nan
        cfg = DataConfig()
        result = handle_missing(df, cfg)
        assert result["EURPLN_CLOSE"].isna().sum() == 0

    def test_ffill_fills_cross_pairs(self):
        df = make_forex_df(n_rows=100, sparse_pairs=["PLNCZK"])
        cfg = DataConfig(ffill_limit=5)
        result = handle_missing(df, cfg)
        # After ffill, consecutive NaNs up to limit should be filled
        consecutive_nan = result["PLNCZK_CLOSE"].isna().sum()
        # Shouldn't be worse than before
        assert consecutive_nan <= df["PLNCZK_CLOSE"].isna().sum()

    def test_returns_reset_index(self):
        df = make_forex_df(n_rows=100)
        cfg = DataConfig()
        result = handle_missing(df, cfg)
        assert list(result.index) == list(range(len(result)))


class TestCreateTarget:
    def test_binary_values(self):
        df = make_forex_df(n_rows=200)
        cfg = DataConfig(horizon_minutes=5)
        target = create_target(df, cfg)
        valid = target.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_last_horizon_rows_are_nan(self):
        df = make_forex_df(n_rows=200)
        cfg = DataConfig(horizon_minutes=5)
        target = create_target(df, cfg)
        assert target.iloc[-5:].isna().all()
        assert target.iloc[:-5].notna().all()

    def test_target_length_matches_df(self):
        df = make_forex_df(n_rows=300)
        cfg = DataConfig()
        target = create_target(df, cfg)
        assert len(target) == len(df)


class TestSplitTimeAware:
    def test_split_sizes(self):
        df = make_forex_df(n_rows=300)
        cfg = DataConfig(horizon_minutes=5)
        target = create_target(df, cfg)

        valid = target.notna()
        X_train, X_test, y_train, y_test = split_time_aware(
            df[valid].reset_index(drop=True),
            target[valid].reset_index(drop=True),
            test_ratio=0.20,
        )
        total = len(y_train) + len(y_test)
        assert abs(len(y_test) / total - 0.20) < 0.02

    def test_no_overlap(self):
        df = make_forex_df(n_rows=300)
        cfg = DataConfig(horizon_minutes=5)
        target = create_target(df, cfg)
        valid = target.notna()
        X = df[valid].reset_index(drop=True)
        y = target[valid].reset_index(drop=True)
        X_train, X_test, y_train, y_test = split_time_aware(X, y)
        # Train timestamps must all precede test timestamps
        assert X_train["timestamp"].max() < X_test["timestamp"].min()
