"""
Evaluation metrics for binary direction classification.

Responsibility: compute and format evaluation metrics.
All functions are pure (no side effects).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute classification metrics for direction prediction.

    Parameters
    ----------
    y_true : ground truth labels (0/1)
    y_pred : predicted labels (0/1)
    y_prob : predicted probabilities for class 1 (optional, for AUC)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # Baseline: how often does the majority class appear?
        "baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def print_cv_summary(summary: dict[str, dict[str, float]]) -> None:
    """Pretty-print walk-forward CV summary."""
    print("\n=== Walk-Forward CV Summary ===")
    print(f"{'Metric':<20} {'Mean':>8} {'Std':>8}")
    print("-" * 38)
    for metric, stats in summary.items():
        print(f"{metric:<20} {stats['mean']:>8.4f} {stats['std']:>8.4f}")
    print()


def feature_importance_df(model, feature_names: list[str]):
    """Return a sorted DataFrame of feature importances."""
    import pandas as pd

    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
