"""
Main pipeline entrypoint.

Usage:
    python run_pipeline.py
    python run_pipeline.py --data data/forex_data.feather --horizon 5

All configuration is centralised in forex.config — edit there, not here.
"""
from __future__ import annotations

import argparse
import functools
import logging
import sys
from pathlib import Path

import pandas as pd

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from forex.config import DataConfig, PipelineConfig
from forex.data.loader import drop_sparse_columns, load_raw
from forex.data.preprocessing import create_target, handle_missing, split_time_aware
from forex.evaluation.metrics import feature_importance_df, print_cv_summary
from forex.features.engineering import build_features
from forex.models.classifier import build_classifier
from forex.training.pipeline import cv_summary, train_final_model, walk_forward_cv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EURPLN direction prediction pipeline")
    parser.add_argument("--data", default=None, help="Path to forex_data.feather")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Prediction horizon in minutes")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation (faster, for debugging)")
    return parser.parse_args()


def run(cfg: PipelineConfig, skip_cv: bool = False) -> None:
    logger.info("=== EURPLN Prediction Pipeline ===")
    logger.info("Target pair: %s | Horizon: %d min", cfg.data.target_pair, cfg.data.horizon_minutes)

    # 1. Load & clean raw data
    df = load_raw(cfg.data)
    df = drop_sparse_columns(df, cfg.data.min_col_coverage)
    df = handle_missing(df, cfg.data)

    # 2. Create target (before feature engineering — avoids contamination)
    target = create_target(df, cfg.data)

    # 3. Build features (index-aligned with df)
    features = build_features(df, cfg.features)

    # Align features, raw df, and target on common index
    common_idx = features.index.intersection(target.dropna().index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]

    logger.info("Aligned dataset: %d samples, %d features", len(X), X.shape[1])

    min_required = cfg.validation.min_train_size + cfg.data.horizon_minutes + 1
    if len(X) < min_required:
        raise ValueError(
            f"Dataset too small after feature warm-up: {len(X)} rows. "
            f"Need at least {min_required} rows "
            f"(min_train_size={cfg.validation.min_train_size} + horizon={cfg.data.horizon_minutes}). "
            f"Ensure your data file contains sufficient history."
        )

    # 4. Temporal train / test split
    X_train, X_test, y_train, y_test = split_time_aware(X, y, test_ratio=0.15)

    # 5. Walk-forward cross-validation on training set
    model_factory = functools.partial(build_classifier, cfg.model)

    if not skip_cv:
        logger.info("Running %d-fold walk-forward CV...", cfg.validation.n_splits)
        fold_results = walk_forward_cv(X_train, y_train, model_factory, cfg.validation)
        summary = cv_summary(fold_results)
        print_cv_summary(summary)
    else:
        logger.info("Skipping CV (--no-cv flag set)")

    # 6. Train final model on full training set
    final_model = train_final_model(X_train, y_train, model_factory)

    # 7. Evaluate on held-out test set
    from forex.evaluation.metrics import compute_metrics
    y_prob_test = final_model.predict_proba(X_test.values)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    test_metrics = compute_metrics(y_test.values, y_pred_test, y_prob_test)

    print("\n=== Hold-Out Test Set Results ===")
    for k, v in test_metrics.items():
        print(f"  {k:<22}: {v:.4f}")

    # 8. Feature importance
    importance = feature_importance_df(final_model, list(X_train.columns))
    print("\n=== Top 20 Features ===")
    print(importance.head(20).to_string(index=False))

    logger.info("Pipeline complete.")
    return final_model, importance


def main() -> None:
    args = parse_args()

    cfg = PipelineConfig()
    if args.data:
        cfg.data.data_path = args.data
    if args.horizon:
        cfg.data.horizon_minutes = args.horizon

    run(cfg, skip_cv=args.no_cv)


if __name__ == "__main__":
    main()
