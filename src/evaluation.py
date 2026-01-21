"""
Evaluation Functions for Foot Traffic Forecasting.

This module contains functions for computing and displaying model metrics.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_metrics_table(y_true, predictions_dict: dict, target_col: str) -> pd.DataFrame:
    """
    Build a small metrics table (per model) with common regression scores.

    Parameters
    ----------
    y_true : array-like or pandas.Series
        Ground-truth targets.
    predictions_dict : dict[str, pandas.DataFrame]
        For each model name, a DataFrame containing a column `target_col`
        with predictions aligned to `y_true`.
    target_col : str
        Name of the prediction column inside each predictions DataFrame.

    Returns
    -------
    pandas.DataFrame
        Rows per model with columns: 'Model', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'.
        Values are **formatted strings** (e.g., "123.45", "12.34%", "0.8765").

    Notes
    -----
    - MAPE uses `np.mean(abs((y_true - y_pred)/y_true))`. If `y_true` contains zeros,
    this will yield inf/NaN; consider filtering or using SMAPE for robustness.
    """
    return pd.DataFrame([{
        'Model': model_name.upper(),
        'MSE': f"{(mean_squared_error(y_true, preds[target_col])):.2f}",
        'RMSE': f"{np.sqrt(mean_squared_error(y_true, preds[target_col])):.2f}",
        'MAE': f"{(mean_absolute_error(y_true, preds[target_col])):.2f}",
        'MAPE': f"{(np.mean(np.abs((y_true - preds[target_col]) / y_true)) * 100):.2f}%",
        'R2': f"{r2_score(y_true, preds[target_col]):.4f}"
    } for model_name, preds in predictions_dict.items()])


def create_street_metrics(val_df: pd.DataFrame, predictions_dict: dict,
                          target_col: str) -> dict:
    """
    Compute the same metrics as `create_metrics_table` **per street**.

    Parameters
    ----------
    val_df : pandas.DataFrame
        Validation frame with columns:
        - 'streetname'
        - `target_col` (ground-truth)
        Must be index-aligned with the predictions in `predictions_dict`.
    predictions_dict : dict[str, pandas.DataFrame]
        For each model name, a DataFrame with a column `target_col` whose index
        matches `val_df.index` (so boolean masks can be applied).
    target_col : str
        Name of the target/prediction column.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from street name -> metrics table (one row per model).

    Notes
    -----
    - Assumes all predictions are aligned to `val_df` so that `loc[street_mask]`
    selects the right rows.
    """
    return {street: create_metrics_table(
        val_df[val_df['streetname'] == street][target_col],
        {model_name: pd.DataFrame({target_col: preds.loc[val_df['streetname'] == street, target_col]})
         for model_name, preds in predictions_dict.items()},
        target_col
    ) for street in val_df['streetname'].unique()}


def display_model_comparison(val_df: pd.DataFrame, val_predictions: dict,
                             test_predictions: dict, best_models: dict) -> None:
    """
    Print overall and per-street validation metrics for all targets, plus the selected best model.

    Parameters
    ----------
    val_df : pandas.DataFrame
        Validation dataframe containing the true targets and a 'streetname' column.
    val_predictions : dict[str, dict[str, pandas.DataFrame]]
        Mapping: target -> (model_name -> predictions_df). Each predictions_df must
        contain the target column with index aligned to `val_df`.
    test_predictions : dict[str, dict[str, pandas.DataFrame]]
        Mapping: target -> (model_name -> predictions_df) for test data.
        (Used only for completeness in this display routine.)
    best_models : dict[str, str]
        Mapping: target -> best model name (e.g., 'xgb').

    Returns
    -------
    None
        Prints formatted tables to stdout.

    Notes
    -----
    Uses `create_metrics_table` for overall metrics and `create_street_metrics`
    for per-street metrics. Assumes targets:
    ['n_pedestrians', 'n_pedestrians_towards', 'n_pedestrians_away'].
    """
    for target in ['n_pedestrians', 'n_pedestrians_towards', 'n_pedestrians_away']:
        print(f"\n{'='*80}\nResults for {target}\n{'='*80}\n")

        print("Overall Metrics:")
        print(create_metrics_table(val_df[target], val_predictions[target], target).to_string(index=False))
        print("\nStreet-wise Metrics:")

        for street, metrics in create_street_metrics(val_df, val_predictions[target], target).items():
            print(f"\n{street}:")
            print(metrics.to_string(index=False))

        if target in best_models:
            print(f"\nBest Model Selected: {best_models[target].upper()}")
        print('-'*80)


def display_feature_importance(model, feature_cols: list, model_name: str,
                               target: str) -> None:
    """
    Print the top 10 feature importances for a fitted model.

    Parameters
    ----------
    model : object
        Fitted model exposing `feature_importances_` (e.g., tree-based models).
    feature_cols : list[str]
        Feature names aligned with the model's input order.
    model_name : str
        Identifier (e.g., 'xgb', 'rf', 'lgb').
    target : str
        Target name for labeling.

    Returns
    -------
    None
        Prints a sorted table (top 10) to stdout.

    Notes
    -----
    If the estimator does not provide `feature_importances_`, this will raise
    an AttributeError.
    """
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 10 Important Features for {model_name.upper()} - {target}:")
    print(importance_df.head(10).to_string(index=False))
    print()


def display_target_metrics(val_df: pd.DataFrame, predictions: dict,
                           target: str) -> None:
    """
    Print overall and per-street validation metrics for a single target.

    Parameters
    ----------
    val_df : pandas.DataFrame
        Validation dataframe with columns:
        - 'streetname'
        - target (true values for this target).
    predictions : dict[str, pandas.DataFrame]
        Mapping: model_name -> predictions_df. Each predictions_df must contain
        a column named `target` with index aligned to `val_df`.
    target : str
        Target column name (e.g., 'n_pedestrians').

    Returns
    -------
    None
        Prints formatted tables to stdout.

    Notes
    -----
    Uses `create_metrics_table` for overall metrics, then computes the same
    metrics per street using mask-based alignment.
    """
    print(f"\n{'='*80}\nResults for {target}\n{'='*80}\n")

    print("Overall Metrics:")
    print(create_metrics_table(val_df[target], predictions, target).to_string(index=False))
    print("\nStreet-wise Metrics:")

    for street in val_df['streetname'].unique():
        print(f"\n{street}:")
        mask = val_df['streetname'] == street
        street_preds = {model: pd.DataFrame({target: preds.loc[mask, target]}) for model, preds in predictions.items()}
        print(create_metrics_table(val_df[mask][target], street_preds, target).to_string(index=False))
