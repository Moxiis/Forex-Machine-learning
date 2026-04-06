"""
Data preprocessing: NaN handling, target creation, train/test splitting.
Responsibility: raw DataFrame → feature-ready DataFrame + target series.
No model logic here.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from forex.config import DataConfig

logger = logging.getLogger(__name__)


def handle_missing(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    Forward-fill sparse cross-pair NaNs (up to ffill_limit rows).
    EURPLN rows with NaN are dropped entirely — they are essential.

    Forward-fill is safe here because in live trading you always have
    the last known price, never a future one.
    """
    target_close = f"{cfg.target_pair}_CLOSE"

    n_before = len(df)
    df = df.dropna(subset=[target_close])
    if len(df) < n_before:
        logger.warning("Dropped %d rows with NaN in %s", n_before - len(df), target_close)

    # Forward-fill auxiliary pair NaNs with a bounded look-back
    df = df.ffill(limit=cfg.ffill_limit)

    remaining_nan_pct = df.isna().mean().mean() * 100
    logger.info("Remaining NaN after ffill: %.2f%%", remaining_nan_pct)
    return df.reset_index(drop=True)


def create_target(df: pd.DataFrame, cfg: DataConfig) -> pd.Series:
    """
    Binary classification target:
        1 → EURPLN closes higher after `horizon_minutes`
        0 → EURPLN closes lower or flat

    Target is aligned to current row (i.e., "what happens horizon rows ahead").
    The last `horizon_minutes` rows will be NaN and must be dropped before training.
    """
    close_col = f"{cfg.target_pair}_CLOSE"
    future_close = df[close_col].shift(-cfg.horizon_minutes)
    target = (future_close > df[close_col]).astype(float)
    target.iloc[-cfg.horizon_minutes:] = np.nan
    return target.rename("target")


def split_time_aware(
    df: pd.DataFrame,
    target: pd.Series,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Simple temporal train/test split — no shuffling.
    `test_ratio` is taken from the chronological end of the dataset.
    """
    valid_mask = target.notna()
    df_clean = df[valid_mask].reset_index(drop=True)
    y_clean = target[valid_mask].reset_index(drop=True)

    cutoff = int(len(df_clean) * (1 - test_ratio))
    X_train, X_test = df_clean.iloc[:cutoff], df_clean.iloc[cutoff:]
    y_train, y_test = y_clean.iloc[:cutoff], y_clean.iloc[cutoff:]

    logger.info(
        "Split → train: %d rows | test: %d rows | class balance train: %.2f%%",
        len(X_train), len(X_test), y_train.mean() * 100,
    )
    return X_train, X_test, y_train, y_test
