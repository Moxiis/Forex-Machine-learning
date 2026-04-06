"""
Central configuration for the EURPLN prediction pipeline.
All tuneable knobs live here — no magic numbers scattered in the code.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data loading and splitting parameters."""
    data_path: str = "data/forex_data.feather"
    target_pair: str = "EURPLN"
    # Predict close price direction N minutes ahead
    horizon_minutes: int = 5
    # Minimum non-NaN ratio per column to keep it
    min_col_coverage: float = 0.30
    # Forward-fill limit for sparse cross-pair NaNs (minutes)
    ffill_limit: int = 5


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    # Lag windows (in rows = minutes) for return features
    lag_windows: list[int] = field(default_factory=lambda: [1, 5, 15, 30, 60])
    # Rolling window sizes for technical indicators
    rolling_windows: list[int] = field(default_factory=lambda: [5, 15, 30, 60])
    # RSI period
    rsi_period: int = 14
    # Bollinger Band period and std multiplier
    bb_period: int = 20
    bb_std: float = 2.0
    # Cross-pair columns to include as auxiliary features (close only)
    aux_pairs: list[str] = field(default_factory=lambda: [
        "EURUSD", "EURCHF", "EURGBP", "EURJPY",
        "PLNUSD", "PLNCHF", "PLNGBP",
    ])


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int | None = 12
    min_samples_leaf: int = 50
    n_jobs: int = -1
    class_weight: str = "balanced"


@dataclass
class ValidationConfig:
    """Walk-forward cross-validation parameters."""
    n_splits: int = 5
    # Minimum training rows per fold
    min_train_size: int = 10_000
    # Gap between train end and test start (prevents leakage from NaN-fill)
    gap: int = 60


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
