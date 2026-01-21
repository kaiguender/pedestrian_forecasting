"""
Modeling Functions for Foot Traffic Forecasting.

This module contains functions for cross-validation, hyperparameter tuning,
model selection, and training utilities.

Main Components
---------------
Cross-Validation:
    get_cv_splits : Build time-ordered CV splits per street

Hyperparameter Tuning:
    tune_model_random : Randomized hyperparameter search
    tune_model_bayes : Bayesian hyperparameter optimization

Model Selection:
    get_best_model : Select model with lowest validation MSE

Training Utilities:
    train_single_model : Train one model with optional tuning
    train_models_for_target : Train all models for a single target variable

Deseasonalization:
    compute_seasonal_factors : Compute mean by (streetname, weekday, hour)
    apply_deseasonalization : Subtract seasonal factors from target
    reverse_deseasonalization : Add seasonal factors back to predictions
"""
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV


def get_cv_splits(df: pd.DataFrame, n_splits: int = 5, len_split: int = 168) -> list:
    """
    Build blocked, **time-ordered** cross-validation splits per street.

    For each split i (0..n_splits-1), the validation window is the last
    `len_split` rows of each street shifted i blocks back in time.
    The training set is everything strictly before that window, again per street.
    Index arrays from the original (reset) DataFrame are returned so they can be
    fed directly into scikit-learn CV routines.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'streetname' column. Assumed to be **chronologically sorted**
        within each street.
    n_splits : int, default 5
        Number of rolling validation windows to produce from the end of the series.
    len_split : int, default 168
        Length of each validation window in rows (e.g., 168 = one week of hourly data).

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of (train_idx, val_idx) tuples (integer indices) for each split.

    Notes
    -----
    - No leakage: validation windows are strictly after the corresponding train windows.
    - If a street has fewer than (i+1)*len_split rows, the split for that street
    will yield an empty train or val segment; ensure your data is long enough.
    """
    df = df.copy()
    df.reset_index(drop=True, inplace=True)

    cv_splits = []

    for i in range(n_splits):
        ids_train = []
        ids_val = []

        for street in df.streetname.unique():
            street_mask = df.streetname == street
            df_street = df[street_mask]

            len_street = df_street.shape[0]

            split_point = len_street - len_split * (i + 1)
            end_point = len_street - len_split * i

            ids_train.append(df_street.iloc[:split_point, :].index)
            ids_val.append(df_street.iloc[split_point:end_point, :].index)

        ids_train = np.concatenate(ids_train, axis=0)
        ids_val = np.concatenate(ids_val, axis=0)

        cv_splits.append((ids_train, ids_val))

    return cv_splits


def tune_model_random(model, param_dist: dict, X, y, cv_splits: list,
                      n_iter: int = 50, n_jobs: int = 8, verbose: int = 1):
    """
    Randomized hyperparameter search with custom time-based splits.

    Wraps `sklearn.model_selection.RandomizedSearchCV` using negative MSE,
    fits on (X, y), and returns the fitted search object.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Estimator supporting `fit` and (optionally) `predict`.
    param_dist : dict or list[dict]
        Parameter distributions for RandomizedSearchCV.
    X : array-like or pandas.DataFrame
        Training features.
    y : array-like or pandas.Series
        Target values aligned with X.
    cv_splits : list[tuple[np.ndarray, np.ndarray]]
        Output from `get_cv_splits` or similar [(train_idx, val_idx), ...].
    n_iter : int, default 50
        Number of parameter settings sampled.
    n_jobs : int, default 8
        Parallel jobs for the CV search.
    verbose : int, default 1
        Verbosity level passed to RandomizedSearchCV.

    Returns
    -------
    sklearn.model_selection.RandomizedSearchCV
        The fitted search object (`best_estimator_`, `best_params_`, etc. available).

    Notes
    -----
    - Scoring is 'neg_mean_squared_error' (larger is better). Use `-cv.best_score_`
    to get the MSE.
    """
    cv = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=cv_splits,
        refit=True,
        n_jobs=n_jobs,
        verbose=verbose)

    cv.fit(X, y)

    return cv


def tune_model_bayes(model, param_dist: dict, X, y, cv_splits: list,
                     n_iter: int = 50, n_points: int = 1, n_jobs: int = 8,
                     verbose: int = 1):
    """
    Bayesian hyperparameter optimization with custom time-based splits.

    Uses `skopt.BayesSearchCV` (scikit-optimize) to search `param_dist`
    under negative MSE, fits on (X, y), and returns the fitted search object.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Estimator supporting `fit`.
    param_dist : dict or list[skopt.space.Dimension]
        Search space for BayesSearchCV.
    X : array-like or pandas.DataFrame
        Training features.
    y : array-like or pandas.Series
        Targets aligned with X.
    cv_splits : list[tuple[np.ndarray, np.ndarray]]
        Output from `get_cv_splits` or similar.
    n_iter : int, default 50
        Total number of parameter evaluations.
    n_points : int, default 1
        Number of parameter settings to sample in parallel per iteration.
    n_jobs : int, default 8
        Parallel jobs for model fitting in each CV fold.
    verbose : int, default 1
        Verbosity level passed to BayesSearchCV.

    Returns
    -------
    skopt.BayesSearchCV
        The fitted search object (`best_estimator_`, `best_params_`, etc.).

    Notes
    -----
    - Scoring is 'neg_mean_squared_error'. Convert to MSE via `-cv.best_score_`.
    - `random_state` is fixed at 42 for reproducibility.
    """
    cv = BayesSearchCV(
        estimator=model,
        search_spaces=param_dist,
        n_iter=n_iter,
        n_points=n_points,
        scoring='neg_mean_squared_error',
        cv=cv_splits,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=42)

    cv.fit(X, y)

    return cv


def get_best_model(models_dict: dict, y_true, val_predictions_dict: dict) -> tuple:
    """
    Pick the model with the lowest MSE on a shared validation target.

    Parameters
    ----------
    models_dict : dict[str, sklearn.base.BaseEstimator]
        Mapping from model name to fitted model (or any identifier).
    y_true : array-like
        Ground-truth validation targets.
    val_predictions_dict : dict[str, array-like or pandas.Series]
        Mapping from model name to its predictions aligned with `y_true`.

    Returns
    -------
    tuple[str, float]
        (best_model_name, best_mse)

    Notes
    -----
    - Assumes `val_predictions_dict[model_name]` is 1-D and index-aligned with `y_true`.
    """
    best_mse, best_model = float('inf'), None

    for model_name, model in models_dict.items():
        mse = mean_squared_error(y_true, val_predictions_dict[model_name])
        if mse < best_mse:
            best_mse, best_model = mse, model_name

    return best_model, best_mse


# =============================================================================
# Training Utilities
# =============================================================================

def train_single_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    feature_cols: List[str],
    search_space: Optional[dict] = None,
    tune: bool = False,
    n_splits: int = 10,
    n_iter: int = 100,
    n_points: int = 5,
    n_jobs: int = 5,
    verbose: int = 1
) -> Tuple[Any, np.ndarray]:
    """
    Train a single model, optionally with Bayesian hyperparameter tuning.

    This function encapsulates the training logic for a single model instance.
    If tuning is enabled, it uses Bayesian optimization via `tune_model_bayes`
    to find optimal hyperparameters before returning the best estimator.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A scikit-learn compatible model instance (e.g., LGBMRegressor, XGBRegressor).
        The model should support `fit(X, y)` and `predict(X)` methods.
    model_name : str
        Human-readable name of the model, used for logging output
        (e.g., 'lgb', 'xgb', 'rf').
    X_train : pd.DataFrame
        Training data containing all columns. The function will select
        only `feature_cols` for training.
    y_train : pd.Series
        Training target values, aligned with X_train index.
    X_val : pd.DataFrame
        Validation data containing all columns. Used for generating
        predictions after training.
    feature_cols : List[str]
        List of column names to use as features. Must be present in
        both X_train and X_val.
    search_space : dict, optional
        Hyperparameter search space for Bayesian optimization.
        Required if `tune=True`. Should use skopt space definitions
        (e.g., Integer, Real). Example:
        {'n_estimators': Integer(100, 1000), 'learning_rate': Real(0.01, 0.3)}
    tune : bool, default False
        If True, perform Bayesian hyperparameter tuning before training.
        Requires `search_space` to be provided.
    n_splits : int, default 10
        Number of cross-validation splits for hyperparameter tuning.
        Only used when `tune=True`.
    n_iter : int, default 100
        Total number of hyperparameter configurations to evaluate.
        Only used when `tune=True`.
    n_points : int, default 5
        Number of hyperparameter configurations to sample in parallel
        per iteration. Only used when `tune=True`.
    n_jobs : int, default 5
        Number of parallel jobs for CV evaluation. Only used when `tune=True`.
    verbose : int, default 1
        Verbosity level for tuning progress. 0=silent, 1=progress,
        2+=detailed. Only used when `tune=True`.

    Returns
    -------
    Tuple[sklearn.base.BaseEstimator, np.ndarray]
        A tuple containing:
        - fitted_model: The trained model (best estimator if tuning was used)
        - val_predictions: Numpy array of predictions on X_val

    Examples
    --------
    >>> from lightgbm import LGBMRegressor
    >>> model = LGBMRegressor(n_estimators=100)
    >>> fitted, preds = train_single_model(
    ...     model=model,
    ...     model_name='lgb',
    ...     X_train=train_df,
    ...     y_train=train_df['target'],
    ...     X_val=val_df,
    ...     feature_cols=['feature1', 'feature2']
    ... )
    """
    print(f"\nTraining {model_name.upper()}...")

    if tune and search_space is not None:
        print(f"Starting hyperparameter tuning for {model_name.upper()}...")
        cv_splits = get_cv_splits(X_train, n_splits=n_splits, len_split=168)
        cv_results = tune_model_bayes(
            model, search_space,
            X_train[feature_cols], y_train,
            cv_splits, n_iter=n_iter, n_points=n_points,
            n_jobs=n_jobs, verbose=verbose
        )
        model = cv_results.best_estimator_
        print("Best hyperparameters:")
        for param, value in cv_results.best_params_.items():
            print(f"  {param:<20}: {value}")
    else:
        model.fit(X_train[feature_cols], y_train)

    val_preds = model.predict(X_val[feature_cols])
    print("Training complete.")

    return model, val_preds


def train_models_for_target(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    models_selected: Dict[str, Any],
    search_spaces: Dict[str, dict],
    tune_models: bool = False,
    retrain_on_val: bool = False,
    n_splits: int = 10,
    n_iter: int = 100,
    n_points: int = 5,
    n_jobs: int = 5,
    verbose: int = 1
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Train all selected models for a single target variable.

    This is the core training loop extracted from the notebook for reusability.
    It orchestrates the complete training workflow for one target:

    1. Iterates over all models in `models_selected`
    2. Trains each model (with optional hyperparameter tuning)
    3. Generates validation predictions
    4. Optionally retrains on combined train+val data
    5. Generates test predictions

    All predictions are clipped to be non-negative (pedestrian counts can't be < 0).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing features and target column. Must include
        'streetname', 'weekday', 'hour' for CV split generation.
    X_val : pd.DataFrame
        Validation data with same structure as X_train. Used for model
        evaluation and generating validation predictions.
    test_df : pd.DataFrame
        Test data for generating final predictions. Must contain all
        columns in `feature_cols`.
    target : str
        Name of the target column (e.g., 'n_pedestrians'). Must be present
        in X_train and X_val.
    feature_cols : List[str]
        List of feature column names to use for training. Must be present
        in X_train, X_val, and test_df.
    models_selected : Dict[str, sklearn.base.BaseEstimator]
        Dictionary mapping model names to model instances. Example:
        {'lgb': LGBMRegressor(...), 'xgb': XGBRegressor(...)}
    search_spaces : Dict[str, dict]
        Hyperparameter search spaces per model name. Keys should match
        `models_selected`. Only used when `tune_models=True`. Example:
        {'lgb': {'n_estimators': Integer(100, 1000)}, 'xgb': {...}}
    tune_models : bool, default False
        If True, perform Bayesian hyperparameter optimization for each model.
    retrain_on_val : bool, default False
        If True, after initial training and validation, retrain each model
        on the combined train+val data before generating test predictions.
        This can improve test performance by using all available labeled data.
    n_splits : int, default 10
        Number of time-based CV splits for hyperparameter tuning.
    n_iter : int, default 100
        Total iterations for Bayesian hyperparameter search.
    n_points : int, default 5
        Parallel evaluations per Bayesian optimization iteration.
    n_jobs : int, default 5
        Parallel jobs for cross-validation.
    verbose : int, default 1
        Verbosity level (0=silent, 1=progress, 2+=detailed).

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, Any]]
        A tuple of three dictionaries, all keyed by model name:

        val_predictions : Dict[str, pd.DataFrame]
            Validation predictions. Each DataFrame has shape (n_val_samples, 1)
            with the target column name and X_val.index as index.

        test_predictions : Dict[str, pd.DataFrame]
            Test predictions. Each DataFrame has shape (n_test_samples, 1)
            with the target column name and test_df.index as index.

        models_fitted : Dict[str, estimator]
            Fitted model objects. These are the models after training
            (and after retraining on val if `retrain_on_val=True`).

    Examples
    --------
    >>> from lightgbm import LGBMRegressor
    >>> from xgboost import XGBRegressor
    >>>
    >>> models = {'lgb': LGBMRegressor(), 'xgb': XGBRegressor()}
    >>> spaces = {'lgb': {...}, 'xgb': {...}}
    >>>
    >>> val_preds, test_preds, fitted = train_models_for_target(
    ...     X_train=train_df,
    ...     X_val=val_df,
    ...     test_df=test_df,
    ...     target='n_pedestrians',
    ...     feature_cols=feature_cols,
    ...     models_selected=models,
    ...     search_spaces=spaces,
    ...     tune_models=False
    ... )
    >>>
    >>> # Access predictions
    >>> lgb_val_preds = val_preds['lgb']
    >>> xgb_test_preds = test_preds['xgb']
    >>> best_model = fitted['lgb']

    Notes
    -----
    - Models are deep-copied before training to avoid modifying the originals
    - Predictions are clipped to non-negative values (a_min=0)
    - The function prints progress messages during training

    See Also
    --------
    train_single_model : Train a single model
    get_cv_splits : Generate time-based CV splits
    tune_model_bayes : Bayesian hyperparameter optimization
    """
    val_predictions = {}
    test_predictions = {}
    models_fitted = {}

    for model_name, model in models_selected.items():
        # Train model (deep copy to avoid modifying the original)
        fitted_model, val_preds = train_single_model(
            model=copy.deepcopy(model),
            model_name=model_name,
            X_train=X_train,
            y_train=X_train[target],
            X_val=X_val,
            feature_cols=feature_cols,
            search_space=search_spaces.get(model_name),
            tune=tune_models,
            n_splits=n_splits,
            n_iter=n_iter,
            n_points=n_points,
            n_jobs=n_jobs,
            verbose=verbose
        )

        # Store validation predictions (clipped to non-negative)
        val_preds = np.clip(val_preds, a_min=0, a_max=None)
        val_predictions[model_name] = pd.DataFrame({target: val_preds}, index=X_val.index)

        # Optionally retrain on train+val for test predictions
        if retrain_on_val:
            combined_df = pd.concat([X_train, X_val], ignore_index=True)
            fitted_model.fit(combined_df[feature_cols], combined_df[target])

        models_fitted[model_name] = fitted_model

        # Test predictions (clipped to non-negative)
        test_preds = fitted_model.predict(test_df[feature_cols])
        test_preds = np.clip(test_preds, a_min=0, a_max=None)
        test_predictions[model_name] = pd.DataFrame({target: test_preds}, index=test_df.index)

    return val_predictions, test_predictions, models_fitted


# =============================================================================
# Deseasonalization Utilities
# =============================================================================

def compute_seasonal_factors(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Compute seasonal factors as mean target value by (streetname, weekday, hour).

    Seasonal factors capture the typical pedestrian traffic pattern for each
    street at each hour of each day of the week. These can be subtracted from
    the target to create a deseasonalized (residual) target for modeling.

    The idea is that a model may learn better from deseasonalized data because
    the strong weekly/hourly patterns are removed, allowing it to focus on
    other signals (weather, events, trends).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the training data. Must have columns:
        - 'streetname': Street identifier
        - 'weekday': Day of week (0=Monday to 6=Sunday)
        - 'hour': Hour of day (0-23)
        - target: The target column to compute seasonal factors for

    target : str
        Name of the target column (e.g., 'n_pedestrians').

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'streetname': Street identifier
        - 'weekday': Day of week
        - 'hour': Hour of day
        - 'seasonal_factor': Mean target value for this (street, weekday, hour)

        Shape will be (n_streets * 7 * 24, 4) assuming all combinations exist.

    Examples
    --------
    >>> seasonal = compute_seasonal_factors(train_df, 'n_pedestrians')
    >>> print(seasonal.head())
       streetname  weekday  hour  seasonal_factor
    0  hauptstrasse       0     0            15.3
    1  hauptstrasse       0     1            12.1
    ...

    See Also
    --------
    apply_deseasonalization : Use these factors to transform data
    reverse_deseasonalization : Add factors back to predictions
    """
    seasonal_factors = (
        df.groupby(['streetname', 'weekday', 'hour'])[[target]]
        .mean()
        .reset_index()
        .rename(columns={target: 'seasonal_factor'})
    )
    return seasonal_factors


def apply_deseasonalization(
    df: pd.DataFrame,
    target: str,
    seasonal_factors: pd.DataFrame
) -> pd.DataFrame:
    """
    Subtract seasonal factors from the target column to deseasonalize.

    This creates a residual target that removes the typical weekly/hourly
    pattern. The model then predicts these residuals, and seasonal factors
    are added back to get final predictions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to transform. Must contain columns:
        - 'streetname': Street identifier
        - 'weekday': Day of week (0-6)
        - 'hour': Hour of day (0-23)
        - target: The target column to deseasonalize

        A copy is made; the original DataFrame is not modified.

    target : str
        Name of the target column to deseasonalize.

    seasonal_factors : pd.DataFrame
        Output from `compute_seasonal_factors()`. Must contain columns:
        - 'streetname', 'weekday', 'hour', 'seasonal_factor'

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with:
        - 'seasonal_factor' column added (merged from seasonal_factors)
        - target column values replaced with (original - seasonal_factor)

    Examples
    --------
    >>> seasonal = compute_seasonal_factors(train_df, 'n_pedestrians')
    >>> train_deseason = apply_deseasonalization(train_df, 'n_pedestrians', seasonal)
    >>> val_deseason = apply_deseasonalization(val_df, 'n_pedestrians', seasonal)

    Notes
    -----
    - The 'seasonal_factor' column is kept in the output so it can be used
      later with `reverse_deseasonalization()`
    - Missing combinations (street, weekday, hour) will get NaN seasonal_factor

    See Also
    --------
    compute_seasonal_factors : Compute the seasonal factors first
    reverse_deseasonalization : Add factors back after prediction
    """
    df = df.copy()
    df = df.merge(seasonal_factors, on=['streetname', 'weekday', 'hour'], how='left')
    df[target] = df[target] - df['seasonal_factor']
    return df


def reverse_deseasonalization(
    predictions: np.ndarray,
    seasonal_factor_values: np.ndarray
) -> np.ndarray:
    """
    Add seasonal factors back to deseasonalized predictions.

    After a model predicts on deseasonalized data, this function restores
    predictions to the original scale by adding back the seasonal factors.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions on the deseasonalized scale. These are the
        residuals predicted by the model.

    seasonal_factor_values : np.ndarray
        Seasonal factor values aligned with the predictions. Typically
        obtained from the 'seasonal_factor' column of a DataFrame that
        was processed with `apply_deseasonalization()`.

    Returns
    -------
    np.ndarray
        Predictions on the original scale (predictions + seasonal_factors).

    Examples
    --------
    >>> # After training on deseasonalized data
    >>> raw_preds = model.predict(X_val[feature_cols])
    >>> final_preds = reverse_deseasonalization(
    ...     raw_preds,
    ...     X_val['seasonal_factor'].values
    ... )
    >>> final_preds = np.clip(final_preds, a_min=0, a_max=None)

    Notes
    -----
    - Remember to clip predictions to non-negative after this operation,
      as pedestrian counts cannot be negative
    - The arrays must be aligned (same length and order)

    See Also
    --------
    compute_seasonal_factors : Compute seasonal factors from training data
    apply_deseasonalization : Apply factors before training
    """
    return predictions + seasonal_factor_values
