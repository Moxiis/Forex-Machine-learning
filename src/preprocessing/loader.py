"""
Data loading and schema validation.
Responsibility: read raw feather/CSV → return a clean, sorted DataFrame.
Does NOT do feature engineering or target creation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from forex.config import DataConfig

logger = logging.getLogger(__name__)

# All 13 currency pairs present in the dataset
EXPECTED_OHLC_PAIRS = [
    "EURPLN", "PLNCZK", "PLNUSD", "PLNCHF", "PLNGBP", "PLNJPY",
    "EURCZK", "EURHUF", "EURUSD", "EURCHF", "EURGBP", "EURJPY", "PLNHUF",
]
OHLC_SUFFIXES = ("_OPEN", "_HIGH", "_LOW", "_CLOSE")


def _expected_columns() -> list[str]:
    return [
        f"{pair}{suffix}"
        for pair in EXPECTED_OHLC_PAIRS
        for suffix in OHLC_SUFFIXES
    ]


def load_raw(cfg: DataConfig) -> pd.DataFrame:
    """
    Load forex data from feather or CSV, validate schema, sort by timestamp.

    Returns
    -------
    pd.DataFrame
        Integer-indexed, sorted ascending by timestamp.
        Columns: timestamp (int64) + OHLC columns (float64).
    """
    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    logger.info("Loading data from %s", path)
    if path.suffix == ".feather":
        df = pd.read_feather(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .feather or .csv")

    _validate_schema(df)

    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise on missing critical columns; warn on unexpected extras."""
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: 'timestamp'")

    missing = [c for c in _expected_columns() if c not in df.columns]
    if missing:
        logger.warning("Missing expected columns (may affect features): %s", missing)

    target_cols = [f"{EXPECTED_OHLC_PAIRS[0]}{s}" for s in OHLC_SUFFIXES]
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"Target pair column missing: {col}")


def drop_sparse_columns(df: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    """
    Drop non-timestamp columns whose non-NaN ratio is below `min_coverage`.
    Always preserves the target pair (EURPLN_*) regardless of coverage.
    """
    target_cols = {c for c in df.columns if c.startswith("EURPLN")}
    coverage = df.notna().mean()
    sparse = [
        c for c in df.columns
        if c != "timestamp" and c not in target_cols and coverage[c] < min_coverage
    ]
    if sparse:
        logger.info("Dropping %d sparse columns (coverage < %.0f%%): %s",
                    len(sparse), min_coverage * 100, sparse)
    return df.drop(columns=sparse)
