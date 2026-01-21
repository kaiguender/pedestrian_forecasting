"""
Foot Traffic Forecasting Pipeline - Source Package

This package contains all the modular components for the forecasting pipeline:
- features: Feature engineering functions
- modeling: Cross-validation, tuning, and model selection
- evaluation: Metrics calculation and display
- visualization: Plotting functions
- uncertainty: Prediction interval computation
"""

from .features import create_all_features, get_feature_columns, align_dataframe_columns
from .modeling import (
    get_cv_splits,
    get_train_val_split_by_date,
    tune_model_bayes,
    tune_model_random,
    get_best_model,
    train_single_model,
    train_models_for_target,
    compute_seasonal_factors,
    apply_deseasonalization,
    reverse_deseasonalization,
)
from .evaluation import (
    create_metrics_table,
    create_street_metrics,
    display_model_comparison,
    display_feature_importance,
    display_target_metrics,
)
from .visualization import plot_model_comparison, plot_prediction_intervals, plot_train_val_predictions
from .uncertainty import compute_prediction_intervals, check_calibration

__all__ = [
    # Features
    'create_all_features',
    'get_feature_columns',
    'align_dataframe_columns',
    # Modeling
    'get_cv_splits',
    'get_train_val_split_by_date',
    'tune_model_bayes',
    'tune_model_random',
    'get_best_model',
    'train_single_model',
    'train_models_for_target',
    'compute_seasonal_factors',
    'apply_deseasonalization',
    'reverse_deseasonalization',
    # Evaluation
    'create_metrics_table',
    'create_street_metrics',
    'display_model_comparison',
    'display_feature_importance',
    'display_target_metrics',
    # Visualization
    'plot_model_comparison',
    'plot_prediction_intervals',
    'plot_train_val_predictions',
    # Uncertainty
    'compute_prediction_intervals',
    'check_calibration',
]
