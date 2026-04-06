"""
Feature engineering for forex time-series prediction.

Responsibility: clean DataFrame → feature matrix (no target, no leakage).

All features are computed using only past data (shift-based / rolling).
No future information leaks in.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from forex.config import FeatureConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Entry point: applies all feature groups in sequence.
    Returns a DataFrame with only feature columns (drops raw OHLC).
    """
    parts: list[pd.DataFrame] = []

    parts.append(_time_features(df))
    parts.append(_eurpln_technical(df, cfg))
    parts.append(_cross_pair_returns(df, cfg))

    features = pd.concat(parts, axis=1)

    n_nan = features.isna().any(axis=1).sum()
    if n_nan:
        logger.info("Dropping %d rows with NaN features (warm-up period)", n_nan)
        features = features.dropna()

    logger.info("Feature matrix: %d rows × %d columns", *features.shape)
    return features


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical encoding of hour-of-day and day-of-week from unix timestamp.
    Avoids discontinuity at midnight/weekend boundaries.
    """
    dt = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    hour = dt.dt.hour + dt.dt.minute / 60.0
    dow = dt.dt.dayofweek.astype(float)

    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        },
        index=df.index,
    )


def _eurpln_technical(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Technical indicators derived solely from EURPLN OHLC.
    All are purely backward-looking.
    """
    close = df["EURPLN_CLOSE"]
    high = df["EURPLN_HIGH"]
    low = df["EURPLN_LOW"]

    feats: dict[str, pd.Series] = {}

    # --- Lagged log-returns ---
    log_close = np.log(close)
    for w in cfg.lag_windows:
        feats[f"eurpln_ret_{w}m"] = log_close.diff(w)

    # --- Rolling statistics ---
    for w in cfg.rolling_windows:
        roll = close.rolling(w)
        feats[f"eurpln_vol_{w}m"] = roll.std()
        feats[f"eurpln_zscore_{w}m"] = (close - roll.mean()) / (roll.std() + 1e-9)

    # --- RSI ---
    feats[f"eurpln_rsi_{cfg.rsi_period}"] = _rsi(close, cfg.rsi_period)

    # --- Bollinger Band %B ---
    feats["eurpln_bb_pct"] = _bb_pct(close, cfg.bb_period, cfg.bb_std)

    # --- ATR (normalised by close) ---
    feats["eurpln_atr_14"] = _atr(high, low, close, 14) / close

    # --- Candle body & wick ratios ---
    open_ = df["EURPLN_OPEN"]
    candle_range = (high - low).replace(0, np.nan)
    feats["eurpln_body_ratio"] = (close - open_).abs() / candle_range
    feats["eurpln_upper_wick"] = (high - close.clip(lower=open_)) / candle_range
    feats["eurpln_lower_wick"] = (close.clip(upper=open_) - low) / candle_range

    return pd.DataFrame(feats, index=df.index)


def _cross_pair_returns(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Lagged log-returns for auxiliary currency pairs.
    Only the CLOSE column is used per pair.
    Pairs missing from df are silently skipped.
    """
    feats: dict[str, pd.Series] = {}
    for pair in cfg.aux_pairs:
        col = f"{pair}_CLOSE"
        if col not in df.columns:
            logger.debug("Auxiliary pair %s not in dataframe, skipping.", col)
            continue
        log_close = np.log(df[col])
        for w in cfg.lag_windows:
            feats[f"{pair.lower()}_ret_{w}m"] = log_close.diff(w)

    return pd.DataFrame(feats, index=df.index)


# ---------------------------------------------------------------------------
# Helper indicator functions
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _bb_pct(close: pd.Series, period: int, n_std: float) -> pd.Series:
    """Bollinger Band %B: position of price within the band (0=lower, 1=upper)."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return (close - lower) / (upper - lower + 1e-9)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()
