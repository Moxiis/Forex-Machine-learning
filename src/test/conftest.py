"""
Shared test fixtures: synthetic forex DataFrames that mirror real schema.
"""
import numpy as np
import pandas as pd
import pytest

PAIRS = [
    "EURPLN", "PLNCZK", "PLNUSD", "PLNCHF", "PLNGBP", "PLNJPY",
    "EURCZK", "EURHUF", "EURUSD", "EURCHF", "EURGBP", "EURJPY", "PLNHUF",
]
SUFFIXES = ["_OPEN", "_HIGH", "_LOW", "_CLOSE"]


def make_forex_df(n_rows: int = 500, seed: int = 42, sparse_pairs: list[str] | None = None) -> pd.DataFrame:
    """
    Create a synthetic forex DataFrame with realistic structure.
    `sparse_pairs` will have 80% NaN values to simulate PLNHUF-style columns.
    """
    rng = np.random.default_rng(seed)

    # Base EURPLN price walk
    base_price = 4.20 + np.cumsum(rng.normal(0, 0.001, n_rows))

    start_ts = 1_700_000_000
    timestamps = np.arange(start_ts, start_ts + n_rows * 60, 60)

    data = {"timestamp": timestamps}

    for pair in PAIRS:
        pair_base = base_price * rng.uniform(0.9, 1.1)
        close = pair_base + np.cumsum(rng.normal(0, 0.001, n_rows))
        spread = np.abs(rng.normal(0, 0.0005, n_rows))

        data[f"{pair}_OPEN"] = close + rng.normal(0, 0.001, n_rows)
        data[f"{pair}_HIGH"] = close + spread
        data[f"{pair}_LOW"] = close - spread
        data[f"{pair}_CLOSE"] = close

    df = pd.DataFrame(data)

    # Inject sparsity for designated pairs
    sparse_pairs = sparse_pairs or ["PLNHUF"]
    for pair in sparse_pairs:
        mask = rng.random(n_rows) < 0.80
        for suffix in SUFFIXES:
            df.loc[mask, f"{pair}{suffix}"] = np.nan

    return df


@pytest.fixture
def forex_df():
    return make_forex_df(n_rows=500)


@pytest.fixture
def small_df():
    return make_forex_df(n_rows=100)
