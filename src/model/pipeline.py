"""
Training pipeline with walk-forward cross-validation.

Responsibility: orchestrate data → features → model fit → collect fold metrics.
Does NOT load data or build features — receives them as arguments (DI).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit

from forex.config import ValidationConfig
from forex.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    metrics: dict[str, float]


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,          # callable() → fresh unfitted estimator
    cfg: ValidationConfig,
) -> list[FoldResult]:
    """
    Walk-forward cross-validation using sklearn TimeSeriesSplit.

    `gap` rows are excluded between each train/test boundary to prevent
    leakage from forward-filled NaN values.

    Returns a list of FoldResult — one per fold.
    """
    tscv = TimeSeriesSplit(
        n_splits=cfg.n_splits,
        gap=cfg.gap,
    )

    results: list[FoldResult] = []
    X_arr, y_arr = X.values, y.values

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_arr), start=1):
        if len(train_idx) < cfg.min_train_size:
            logger.info("Fold %d skipped: train size %d < min_train_size %d",
                        fold, len(train_idx), cfg.min_train_size)
            continue
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        fold_result = FoldResult(
            fold=fold,
            train_size=len(train_idx),
            test_size=len(test_idx),
            metrics=metrics,
        )
        results.append(fold_result)
        logger.info(
            "Fold %d | train=%d test=%d | acc=%.4f auc=%.4f f1=%.4f",
            fold, len(train_idx), len(test_idx),
            metrics["accuracy"], metrics["roc_auc"], metrics["f1"],
        )

    return results


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_factory,
) -> ClassifierMixin:
    """Fit the final model on the full training set."""
    model = model_factory()
    model.fit(X_train.values, y_train.values)
    logger.info("Final model trained on %d samples", len(y_train))
    return model


def cv_summary(results: list[FoldResult]) -> dict[str, dict[str, float]]:
    """Aggregate fold metrics: mean ± std for each metric."""
    all_metrics = [r.metrics for r in results]
    keys = all_metrics[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in all_metrics])),
            "std": float(np.std([m[k] for m in all_metrics])),
        }
        for k in keys
    }
