"""
Uncertainty Quantification for Foot Traffic Forecasting.

This module contains functions for computing prediction intervals
and checking their calibration.

Prediction intervals from error quantiles -- what we're doing and why:

Goal: We want an uncertainty band around each point forecast that captures
the true value with a specified probability (e.g., 90% or 95%).

Approach:
1. Compute residuals (actual - predicted) on training data
2. Group residuals by relevant strata (e.g., weekday x hour x street)
3. Compute quantiles of residuals within each stratum
4. For new predictions, add the corresponding error quantiles to get
   prediction intervals

This is a non-parametric approach that captures heteroscedasticity
(varying uncertainty by time of day, day of week, etc.).
"""
import numpy as np
import pandas as pd
from typing import List


def compute_prediction_intervals(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    feature_cols: List[str],
    target: str,
    quantiles: List[float] = None,
    cols_to_keep: List[str] = None,
) -> tuple:
    """
    Compute prediction intervals using error quantiles from training data.

    For each street, computes the error (actual - predicted) on training data,
    then calculates quantiles of those errors grouped by (weekday, hour).
    These quantiles are then applied to validation and test predictions.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training data with features and target. Must contain 'streetname',
        'weekday', 'hour', and target column.
    val_df : pandas.DataFrame
        Validation data with same structure as train_df.
    test_df : pandas.DataFrame
        Test data with same structure (target may be missing).
    model : fitted model
        Model with a .predict() method.
    feature_cols : List[str]
        Feature column names to use for prediction.
    target : str
        Target column name.
    quantiles : List[float], optional
        Quantiles to compute. Default is a comprehensive set from 0.005 to 0.995.
    cols_to_keep : List[str], optional
        Columns to keep in output. Default is ['id', 'streetname', 'date', 'hour', 'weekday'].

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (pred_results_train, pred_results_val, pred_results_test)
        Each DataFrame contains:
        - cols_to_keep columns
        - 'target': name of the target variable
        - 'actual': true values (NaN for test if not available)
        - 'pred_mean': point prediction
        - 'quantile': quantile level
        - 'error_quantile': error quantile value
        - 'pred_quantile': prediction interval bound
        - 'error_pred_quantile': actual - pred_quantile (for calibration)
    """
    if quantiles is None:
        quantiles = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995]

    if cols_to_keep is None:
        cols_to_keep = ['id', 'streetname', 'date', 'hour', 'weekday']

    pred_results_train = []
    pred_results_val = []
    pred_results_test = []

    for street in train_df['streetname'].unique():
        # --- Training data: compute error quantiles ---
        X_train_street = train_df[train_df['streetname'] == street].copy()

        X_train_street['pred_mean'] = np.clip(model.predict(X_train_street[feature_cols]), a_min=0, a_max=None)
        X_train_street['error'] = X_train_street[target] - X_train_street['pred_mean']

        # Calculate the quantiles of the column error per hour and weekday
        error_quantiles = (X_train_street
                          .groupby(['weekday', 'hour'])['error']
                          .quantile(quantiles)
                          .unstack()
                          .reset_index())
        error_quantiles = error_quantiles.melt(
            id_vars=['weekday', 'hour'],
            value_vars=quantiles,
            var_name='quantile',
            value_name='error_quantile'
        )

        # --- Process training results ---
        pred_results_train_street = X_train_street[cols_to_keep + [target] + ['pred_mean']].merge(
            error_quantiles, on=['weekday', 'hour'], how='right'
        )
        pred_results_train_street.reset_index(drop=True, inplace=True)
        pred_results_train_street['pred_quantile'] = np.clip(
            pred_results_train_street['pred_mean'] + pred_results_train_street['error_quantile'],
            a_min=0, a_max=None
        )
        pred_results_train_street['target'] = target
        pred_results_train_street.rename({target: 'actual'}, axis=1, inplace=True)
        pred_results_train_street['error_pred_quantile'] = (
            pred_results_train_street['actual'] - pred_results_train_street['pred_quantile']
        )
        pred_results_train.append(pred_results_train_street)

        # --- Validation data ---
        X_val_street = val_df[val_df['streetname'] == street].copy()
        X_val_street['pred_mean'] = np.clip(model.predict(X_val_street[feature_cols]), a_min=0, a_max=None)

        pred_results_val_street = X_val_street[cols_to_keep + [target] + ['pred_mean']].merge(
            error_quantiles, on=['weekday', 'hour'], how='left'
        )
        pred_results_val_street.reset_index(drop=True, inplace=True)
        pred_results_val_street['pred_quantile'] = np.clip(
            pred_results_val_street['pred_mean'] + pred_results_val_street['error_quantile'],
            a_min=0, a_max=None
        )
        pred_results_val_street['target'] = target
        pred_results_val_street.rename({target: 'actual'}, axis=1, inplace=True)
        pred_results_val_street['error_pred_quantile'] = (
            pred_results_val_street['actual'] - pred_results_val_street['pred_quantile']
        )
        pred_results_val.append(pred_results_val_street)

        # --- Test data ---
        X_test_street = test_df[test_df['streetname'] == street].copy()
        X_test_street['pred_mean'] = np.clip(model.predict(X_test_street[feature_cols]), a_min=0, a_max=None)

        # Handle case where target may not exist in test data
        test_cols = cols_to_keep + ['pred_mean']
        if target in X_test_street.columns:
            test_cols = cols_to_keep + [target] + ['pred_mean']

        pred_results_test_street = X_test_street[test_cols].merge(
            error_quantiles, on=['weekday', 'hour'], how='left'
        )
        pred_results_test_street.reset_index(drop=True, inplace=True)
        pred_results_test_street['pred_quantile'] = np.clip(
            pred_results_test_street['pred_mean'] + pred_results_test_street['error_quantile'],
            a_min=0, a_max=None
        )
        pred_results_test_street['target'] = target

        if target in X_test_street.columns:
            pred_results_test_street.rename({target: 'actual'}, axis=1, inplace=True)
            pred_results_test_street['error_pred_quantile'] = (
                pred_results_test_street['actual'] - pred_results_test_street['pred_quantile']
            )
        else:
            pred_results_test_street['actual'] = np.nan
            pred_results_test_street['error_pred_quantile'] = np.nan

        pred_results_test.append(pred_results_test_street)

    # Concatenate results
    pred_results_train = pd.concat(pred_results_train, ignore_index=True)
    pred_results_val = pd.concat(pred_results_val, ignore_index=True)
    pred_results_test = pd.concat(pred_results_test, ignore_index=True)

    return pred_results_train, pred_results_val, pred_results_test


def check_calibration(pred_results: pd.DataFrame) -> pd.DataFrame:
    """
    Check if prediction intervals are well calibrated.

    For each quantile, computes the empirical coverage: the fraction of times
    the actual value is below the predicted quantile. For a well-calibrated
    model, this should equal the nominal quantile level.

    For example, for the 95% quantile, the empirical coverage should be at
    least 95% (i.e., the actual value is below the predicted 95% quantile
    at least 95% of the time).

    Parameters
    ----------
    pred_results : pandas.DataFrame
        Output from compute_prediction_intervals containing:
        - 'target': target variable name
        - 'quantile': quantile level
        - 'error_pred_quantile': actual - pred_quantile

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - 'target': target variable
        - 'quantile': nominal quantile level
        - 'empirical_coverage': actual fraction of values below quantile

    Notes
    -----
    If empirical_coverage < quantile for upper quantiles (e.g., 0.95),
    the intervals are too narrow (overconfident).
    If empirical_coverage > quantile, the intervals are too wide (conservative).
    """
    calibration = pred_results.groupby(['target', 'quantile'])['error_pred_quantile'].agg(
        lambda x: np.mean(x < 0)
    ).reset_index()
    calibration.columns = ['target', 'quantile', 'empirical_coverage']

    return calibration
