"""
Model definitions.
Responsibility: wrap sklearn estimators with a stable interface.
Swap the underlying estimator here without touching training/evaluation code.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from forex.config import ModelConfig


def build_classifier(cfg: ModelConfig) -> RandomForestClassifier:
    """
    Baseline: RandomForestClassifier.
    - No scaling required (tree-based).
    - `class_weight='balanced'` compensates for slight directional imbalance.
    - `min_samples_leaf` prevents over-fitting on noisy 1-minute ticks.
    """
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight=cfg.class_weight,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )
