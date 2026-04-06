"""Tests for training pipeline and evaluation metrics."""
import functools
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forex.config import ModelConfig, ValidationConfig
from forex.evaluation.metrics import compute_metrics
from forex.models.classifier import build_classifier
from forex.training.pipeline import cv_summary, train_final_model, walk_forward_cv

from tests.conftest import make_forex_df


def _make_Xy(n: int = 2000, seed: int = 0):
    """Simple synthetic feature matrix and binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 10)), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(rng.integers(0, 2, n).astype(float))
    return X, y


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([0, 1, 0, 1, 1])
        m = compute_metrics(y, y, y.astype(float))
        assert m["accuracy"] == 1.0
        assert m["roc_auc"] == 1.0

    def test_keys_present(self):
        y = np.array([0, 1, 0, 1])
        pred = np.array([0, 1, 1, 0])
        m = compute_metrics(y, pred, pred.astype(float))
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "baseline_accuracy"]:
            assert k in m

    def test_no_crash_on_single_class(self):
        """All-zero target edge case."""
        y = np.zeros(50)
        pred = np.zeros(50)
        m = compute_metrics(y, pred)
        assert "accuracy" in m


class TestWalkForwardCV:
    def test_returns_correct_fold_count(self):
        X, y = _make_Xy(3000)
        cfg = ValidationConfig(n_splits=3, min_train_size=500, gap=10)
        factory = functools.partial(build_classifier, ModelConfig(n_estimators=10))
        results = walk_forward_cv(X, y, factory, cfg)
        # Some folds may be skipped if train size < min_train_size
        assert 1 <= len(results) <= 3

    def test_fold_metrics_are_floats(self):
        X, y = _make_Xy(3000)
        cfg = ValidationConfig(n_splits=2, min_train_size=500, gap=10)
        factory = functools.partial(build_classifier, ModelConfig(n_estimators=5))
        results = walk_forward_cv(X, y, factory, cfg)
        for fold in results:
            for v in fold.metrics.values():
                assert isinstance(v, float)

    def test_train_size_grows_across_folds(self):
        X, y = _make_Xy(3000)
        cfg = ValidationConfig(n_splits=3, min_train_size=500, gap=5)
        factory = functools.partial(build_classifier, ModelConfig(n_estimators=5))
        results = walk_forward_cv(X, y, factory, cfg)
        sizes = [r.train_size for r in results]
        assert sizes == sorted(sizes), "Training set must grow in walk-forward CV"


class TestTrainFinalModel:
    def test_model_predicts_proba(self):
        X, y = _make_Xy(500)
        factory = functools.partial(build_classifier, ModelConfig(n_estimators=10))
        model = train_final_model(X, y, factory)
        proba = model.predict_proba(X.values)
        assert proba.shape == (500, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestCvSummary:
    def test_summary_keys(self):
        X, y = _make_Xy(2000)
        cfg = ValidationConfig(n_splits=2, min_train_size=500, gap=5)
        factory = functools.partial(build_classifier, ModelConfig(n_estimators=5))
        results = walk_forward_cv(X, y, factory, cfg)
        summary = cv_summary(results)
        for metric_stats in summary.values():
            assert "mean" in metric_stats
            assert "std" in metric_stats
