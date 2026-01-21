"""
Visualization Functions for Foot Traffic Forecasting.

This module contains all plotting functions for model comparison
and prediction interval visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_model_comparison(val_df: pd.DataFrame, predictions_dict: dict,
                          test_df: pd.DataFrame, test_predictions_dict: dict,
                          val_test_actuals: pd.DataFrame, target_col: str) -> None:
    """
    Plot actuals and model predictions over time for each street (validation + test).

    Produces one subplot per street. Each subplot shows:
    - Validation actuals (solid blue)
    - Validation+test actuals overlay (dashed blue) from `val_test_actuals`
    - Model predictions on validation (dashed, colored by model)
    - Model predictions on test (dotted, colored by model)
    - A vertical line marking the start of the test period

    Parameters
    ----------
    val_df : pandas.DataFrame
        Validation data with columns: 'streetname', 'datetime', target_col.
    predictions_dict : dict[str, pandas.DataFrame]
        Mapping: model_name -> predictions on validation (must have target_col, index aligned to val_df).
    test_df : pandas.DataFrame
        Test data with columns: 'streetname', 'datetime', target_col (if available for display).
    test_predictions_dict : dict[str, pandas.DataFrame]
        Mapping: model_name -> predictions on test (must have target_col, index aligned to test_df).
    val_test_actuals : pandas.DataFrame
        Frame containing actuals for the combined val/test period with columns:
        'streetname', 'datetime', target_col. (If not fully available, plot will still run.)
    target_col : str
        Target column to plot (e.g., 'n_pedestrians').

    Returns
    -------
    None
        Displays the matplotlib figure.

    Notes
    -----
    - Colors for models are mapped via: {'rf': 'red', 'lgb': 'green', 'xgb': 'purple'}.
    If you use other model keys, extend the mapping or let matplotlib auto-assign.
    - Assumes `val_df` and `test_df` are time-sorted per street.
    """
    n_streets = len(val_df['streetname'].unique())
    fig, axes = plt.subplots(n_streets, 1, figsize=(20, 8*n_streets))
    if n_streets == 1:
        axes = [axes]
    colors = {'rf': 'red', 'lgb': 'green', 'xgb': 'purple'}

    for ax, street in zip(axes, val_df['streetname'].unique()):
        street_mask = val_df['streetname'] == street
        ax.plot(val_df[street_mask]['datetime'], val_df[street_mask][target_col],
                label='Actual', color='blue', alpha=0.7)

        val_test_mask = val_test_actuals['streetname'] == street
        ax.plot(val_test_actuals[val_test_mask]['datetime'],
                val_test_actuals.loc[val_test_mask, target_col],
                label='Actual (Val, Test)', color='blue', linestyle='--', alpha=0.7)

        for model_name in predictions_dict.keys():
            ax.plot(val_df[street_mask]['datetime'],
                    predictions_dict[model_name].loc[street_mask, target_col],
                    label=f'{model_name.upper()} (Train)',
                    color=colors.get(model_name, None), linestyle='--', alpha=0.7)

            test_mask = test_df['streetname'] == street
            ax.plot(test_df[test_mask]['datetime'],
                    test_predictions_dict[model_name].loc[test_mask, target_col],
                    label=f'{model_name.upper()} (Val, Test)',
                    color=colors.get(model_name, None), linestyle=':', alpha=0.7)

        min_test_date = test_df['datetime'].min()
        ax.axvline(x=min_test_date, color='black', linestyle='-', alpha=0.3)
        ax.text(min_test_date, ax.get_ylim()[1], 'Test Period Start',
                rotation=90, verticalalignment='top')

        if target_col == 'n_pedestrians':
            ax.set_title(f"Number of Pedestrians - {street.capitalize()}", fontsize=18)
        elif target_col == 'n_pedestrians_towards':
            ax.set_title(f"Number of Pedestrians Towards - {street.capitalize()}", fontsize=18)
        else:
            ax.set_title(f"Number of Pedestrians Away - {street.capitalize()}", fontsize=18)

        ax.set_xlabel('Date', fontsize=16)
        ax.set_ylabel('Number of Pedestrians', fontsize=16)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        ax.grid(True)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_prediction_intervals(df: pd.DataFrame,
                              target: str,
                              street: str,
                              lower_q: float = 0.05,
                              upper_q: float = 0.95,
                              date_col: str = "date",
                              hour_col: str = "hour",
                              title: str = None) -> None:
    """
    Plot prediction interval [lower_q, upper_q], mean prediction, and actuals
    for a given target and street.

    If actuals are not available, only the prediction interval and mean prediction are shown.

    IMPORTANT: The function is only suited to plot short time frames (e.g. a few weeks).
    For longer time frames, consider resampling the data (e.g. daily averages).

    Assumes df has multiple rows per timestamp (one per quantile). We aggregate:
      - quantile predictions: mean per (timestamp, quantile)
      - pred_mean / actual: mean per timestamp

    Required columns in df:
      'streetname', 'target', date_col, hour_col,
      'actual', 'pred_mean', 'pred_quantile', 'quantile'

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing prediction results with quantile information.
    target : str
        Target variable name (e.g., 'n_pedestrians').
    street : str
        Street name to filter and plot.
    lower_q : float, default 0.05
        Lower quantile for the prediction interval.
    upper_q : float, default 0.95
        Upper quantile for the prediction interval.
    date_col : str, default "date"
        Name of the date column.
    hour_col : str, default "hour"
        Name of the hour column.
    title : str, optional
        Custom title for the plot. If None, auto-generated.

    Returns
    -------
    None
        Displays the matplotlib figure.
    """
    # Filter to one target & street
    df = df[(df["target"] == target) & (df["streetname"] == street)].copy()
    if df.empty:
        raise ValueError(f"No rows for target='{target}' and street='{street}'.")

    # Ensure the actual column exists. It is ignored if not available.
    if "actual" not in df.columns:
        df["actual"] = np.nan  # Placeholder if actuals not available

    # Build timestamp
    df["timestamp"] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[hour_col], unit="h")

    # --- Aggregate so indices are unique ---
    # 1) Quantiles as columns (average in case of duplicates)
    df_quant = df.pivot_table(index="timestamp",
                              columns="quantile",
                              values="pred_quantile",
                              aggfunc="mean")

    # 2) Mean prediction and actuals per timestamp (average if duplicates)
    df_mean = df.groupby("timestamp")[["pred_mean", "actual"]].mean()

    # Check if requested quantiles exist
    if lower_q not in df_quant.columns or upper_q not in df_quant.columns:
        available = sorted(df_quant.columns.tolist())
        raise ValueError(f"Quantiles {lower_q} and/or {upper_q} not found. Available: {available}")

    lower = df_quant[lower_q]
    upper = df_quant[upper_q]
    pred_mean = df_mean["pred_mean"]
    actual = df_mean["actual"]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(df_quant.index, lower, upper, alpha=0.3, label=f"{int((upper_q-lower_q)*100)}% PI")
    ax.plot(df_quant.index, pred_mean, color="blue", label="Prediction (mean)")

    if actual.notna().any():
        ax.plot(df_quant.index, actual, color="black", linestyle="--", label="Actual")

    ax.set_xlabel("Time")
    ax.set_ylabel(target)
    ax.set_title(title or f"{target} - {street}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
