"""
Feature Engineering Functions for Foot Traffic Forecasting.

This module contains all functions for creating features from the raw data,
including time-based features, weather features, event features, and more.
"""
import pandas as pd
import numpy as np
from typing import Sequence
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('..')
from config import GENERAL_DATA_PATH


def create_base_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic calendar and cyclic time features.

    Builds a naive hourly `datetime` from `date` and `hour`, then derives
    calendar fields and cyclic encodings.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        - 'date' (string/datetime-like, date at day resolution)
        - 'hour' (int, 0-23 or 1-24; interpreted as hours since `date`)

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with added columns:
        - 'datetime' (pd.Timestamp)
        - 'year', 'month', 'day' (int)
        - 'day_of_week' (int, Monday=0)
        - 'hour_of_week' (int, 0..167)
        - '{hour,day_of_week,month}_{sin,cos}' (float)

    Notes
    -----
    The hour is added as `pd.to_timedelta(df['hour'], 'h')`. Ensure your 'hour'
    column matches that convention (0-23 recommended).
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df['year'], df['month'], df['day'] = df['datetime'].dt.year, df['datetime'].dt.month, df['datetime'].dt.day
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_of_week'] = df['day_of_week'] * 24 + df['hour']

    for unit, period in [('hour', 24), ('day_of_week', 7), ('month', 12)]:
        df[f'{unit}_sin'] = np.sin(2 * np.pi * df[unit]/period)
        df[f'{unit}_cos'] = np.cos(2 * np.pi * df[unit]/period)

    return df


def create_time_block_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary indicators for commonly used intraday/weekday blocks.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - 'day_of_week' (int, 0=Mon..6=Sun)
        - 'hour' (int, 0-23)

    Returns
    -------
    pandas.DataFrame
        Input `df` with integer {0,1} columns:
        - is_weekend, is_peak_day, is_morning_rush, is_evening_rush,
        is_rush_hour, is_shopping_hours, is_working_hours, is_lunch_time,
        is_night, is_tourist_hours
    """
    time_blocks = {
        'is_weekend': df['day_of_week'].isin([5, 6]),
        'is_peak_day': df['day_of_week'].isin([2,3,4]),
        'is_morning_rush': (df['hour'].between(7, 9)),
        'is_evening_rush': (df['hour'].between(16, 18)),
        'is_rush_hour': (df['hour'].between(7, 9) | df['hour'].between(16, 18)),
        'is_shopping_hours': (df['hour'].between(10, 19)),
        'is_working_hours': (df['hour'].between(9, 17)),
        'is_lunch_time': (df['hour'].between(11, 14)),
        'is_night': ((df['hour'] >= 22) | (df['hour'] <= 5)),
        'is_tourist_hours': (df['hour'].between(10, 18))
    }

    for name, condition in time_blocks.items():
        df[name] = condition.astype(int)
    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather condition one-hots and temperature-derived features.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - 'weather_condition' (object / category)
        - 'temperature' (numeric)

    Returns
    -------
    pandas.DataFrame
        Input `df` with added columns:
        - 'weather_*' one-hot dummies for `weather_condition` (0/1)
        - 'temp_squared' (float), 'temp_norm' (float)
        - 'temp_band' (category: cold/mild/warm/hot)
        - 'temp_*' one-hot dummies for 'temp_band' (0/1)
    """
    df = pd.concat([df, pd.get_dummies(df['weather_condition'], prefix='weather').astype(int)], axis=1)
    df['temp_squared'], df['temp_norm'] = df['temperature'] ** 2, (df['temperature'] - 15) / 10

    df['temp_band'] = pd.cut(df['temperature'], bins=[-np.inf, 5, 15, 25, np.inf], labels=['cold', 'mild', 'warm', 'hot'])
    df = pd.concat([df, pd.get_dummies(df['temp_band'], prefix='temp').astype(int)], axis=1)

    return df


def add_wuerzburg_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge city event and lecture indicators for Würzburg onto the hourly frame.

    Reads daily/hourly event datasets from disk, derives 'date' and 'hour',
    merges them, and flags university exam periods.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - 'date' (string/datetime-like)
        - 'hour' (int)
        - 'month' (int)  # used for 'is_exam_period'

    Returns
    -------
    pandas.DataFrame
        `df` with columns from the merged event datasets and:
        - 'is_exam_period' (0/1) for months {1,2,7,8}

    Notes
    -----
    Uses GENERAL_DATA_PATH from config for file locations.
    """
    eventsDf = pd.read_csv(GENERAL_DATA_PATH / 'events_daily.csv')
    lecturesDf = pd.read_csv(GENERAL_DATA_PATH / 'lectures_daily.csv')

    # Split up date in eventsDf into date and hour
    eventsDf['hour'] = pd.to_datetime(eventsDf['date']).dt.hour.astype('int64')
    eventsDf['date'] = pd.to_datetime(eventsDf['date']).dt.date.astype('str')

    df = df.merge(eventsDf, on=['date', 'hour'], how='left')
    df = df.merge(lecturesDf, on='date', how='left')

    df['is_exam_period'] = (df['month'].isin([1,2,7,8])).astype(int)

    return df


def add_enhanced_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge public/school holidays and derive bridge-day and nationwide flags.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - 'date' (string/datetime-like)
        - 'is_weekend' (0/1)  # used for bridge-day detection

    Returns
    -------
    pandas.DataFrame
        `df` with added/renamed columns:
        - 'is_public_holiday' (int), 'nationwide' (as provided), 'is_public_holiday_nationwide' (boolean/int)
        - 'is_school_holiday' (int)
        - 'is_bridge_day' (0/1), true when a holiday neighbors a weekend.

    Notes
    -----
    Uses GENERAL_DATA_PATH from config for file locations.
    Uses `.shift(+/-1)` across the full DataFrame chronology to detect bridge days.
    """
    publicHolidaysDf = pd.read_csv(GENERAL_DATA_PATH / 'bavarian_public_holidays_daily.csv')
    schoolHolidaysDf = pd.read_csv(GENERAL_DATA_PATH / 'bavarian_school_holidays_daily.csv')

    df = df.merge(publicHolidaysDf, on='date', how='left')

    df['is_bridge_day'] = (
        ((df['public_holiday'].shift(1) == 1) & (df['is_weekend'] == 1)) |
        ((df['public_holiday'] == 1) & (df['is_weekend'].shift(-1) == 1))
    ).astype(int)

    df['is_public_holiday_nationwide'] = (df['public_holiday'] & df['nationwide'])

    df = df.merge(schoolHolidaysDf, on='date', how='left')

    # Rename columns for clarity
    df.rename(columns={'public_holiday': 'is_public_holiday', 'school_holiday': 'is_school_holiday'}, inplace=True)

    return df


def add_timestamp(df: pd.DataFrame, date_col: str = "date", hour_col: str = "hour") -> pd.DataFrame:
    """
    Construct a naive hourly timestamp from separate date and hour columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain `date_col` and `hour_col`.
    date_col : str, default "date"
        Column with date-like values (day resolution).
    hour_col : str, default "hour"
        Column with integer hours; interpreted as hours since `date`.

    Returns
    -------
    pandas.DataFrame
        `df` with an added 'timestamp' (pd.Timestamp) column.

    Notes
    -----
    The hour is treated as `0..` hours offset from `date`. If your data uses 1-24,
    ensure consistency with prior feature builders.
    """
    ts = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[hour_col] - 1, unit="h")
    return df.assign(timestamp=ts)


def add_rolling_lag_means(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    windows_hours: Sequence[int] = (24, 48, 168),
    timestamp_col: str = "timestamp",
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Compute time-based rolling lag means per street, excluding the current hour.

    For each column in `value_cols`, adds `<col>_mean_lag{w}h` using a
    time-based window of the last `w` hours relative to each timestamp,
    applied **per streetname**, and based only on available rows within the
    window (gaps allowed). The current hour is excluded via `shift(1)`.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - `timestamp_col` (datetime-like)
        - 'streetname' (object)
        - all `value_cols` (numeric)
    value_cols : Sequence[str]
        Columns to aggregate.
    windows_hours : Sequence[int], default (24, 48, 168)
        Window sizes in hours (e.g., 24 for last day).
    timestamp_col : str, default "timestamp"
        Name of the timestamp column to use as index for rolling.
    min_periods : int, default 1
        Minimum number of observations in the window to compute a mean.

    Returns
    -------
    pandas.DataFrame
        DataFrame sorted by timestamp with added lag-mean columns.
        Initial NA rows (first observations per street) are dropped.
        Index is reset.

    Notes
    -----
    - Excludes the current row via `.shift(1)`.
    - Uses `'wH'` time-based windows (not fixed-size row counts).
    - Drops all rows with any NA after feature creation (`dropna()`).
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing '{timestamp_col}'. Call add_timestamp() first or pass the right column.")

    out = df.sort_values(timestamp_col).copy()
    out = out.set_index(timestamp_col)

    data_out = []
    for street in out['streetname'].unique():
        street_mask = out['streetname'] == street
        street_data = out[street_mask].copy()
        for col in value_cols:
            s = street_data[col].shift(1)  # exclude current hour
            for w in windows_hours:
                street_data[f"{col}_mean_lag{w}h"] = s.rolling(f"{w}h", min_periods=min_periods).mean()
        data_out.append(street_data)

    out = pd.concat(data_out).sort_index()

    # Remove all NA rows that could not be filled (e.g. first hour of each street)
    out = out.dropna().reset_index(drop=True)

    return out


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add selected pairwise interaction features.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - 'temperature', 'hour', 'is_weekend', 'is_shopping_hours', 'is_rush_hour'
        - 'weather_condition' (used to check for 'rain')

    Returns
    -------
    pandas.DataFrame
        Input `df` with added integer {0,1} columns:
        - 'temp_hour', 'weekend_hour', 'temp_shopping_hours',
        'rain_rush_hour', 'rain_weekend'
    """
    interactions = {
        'temp_hour': df['temperature'] * df['hour'],
        'weekend_hour': df['is_weekend'] * df['hour'],
        'temp_shopping_hours': df['temperature'] * df['is_shopping_hours'],
        'rain_rush_hour': (df['weather_condition'] == 'rain') & df['is_rush_hour'],
        'rain_weekend': (df['weather_condition'] == 'rain') & df['is_weekend']
    }

    for name, interaction in interactions.items():
        df[name] = interaction.astype(int)
    return df


def add_street_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode street names and add street-specific time block interactions.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - 'streetname' (object)
        - 'is_shopping_hours', 'is_rush_hour' (0/1)

    Returns
    -------
    pandas.DataFrame
        Input `df` with:
        - 'street_*' one-hot dummies (0/1)
        - street x shopping/rush interaction flags for
        {'kaiserstrasse','schoenbornstrasse','spiegelstrasse'}
    """
    df = pd.concat([df, pd.get_dummies(df['streetname'], prefix='street').astype(int)], axis=1)

    df['is_kaiserstrasse_shopping'] = ((df['streetname'] == 'kaiserstrasse') & (df['is_shopping_hours'] == 1)).astype(int)
    df['is_schoenbornstrasse_shopping'] = ((df['streetname'] == 'schoenbornstrasse') & (df['is_shopping_hours'] == 1)).astype(int)
    df['is_spiegelstrasse_shopping'] = ((df['streetname'] == 'spiegelstrasse') & (df['is_shopping_hours'] == 1)).astype(int)

    df['is_kaiserstrasse_rush'] = ((df['streetname'] == 'kaiserstrasse') & (df['is_rush_hour'] == 1)).astype(int)
    df['is_schoenbornstrasse_rush'] = ((df['streetname'] == 'schoenbornstrasse') & (df['is_rush_hour'] == 1)).astype(int)
    df['is_spiegelstrasse_rush'] = ((df['streetname'] == 'spiegelstrasse') & (df['is_rush_hour'] == 1)).astype(int)

    return df


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add coarse COVID period flags and seasonal/tourism indicators.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - 'date' (string/period-like, compared as strings 'YYYY-MM')
        - 'month' (int)
        - 'is_weekend' (0/1)

    Returns
    -------
    pandas.DataFrame
        Input `df` with added columns:
        - 'covid_lockdown', 'covid_lockdown_lift', 'covid_lull', 'post_covid_recovery' (0/1)
        - 'season' (category: winter/spring/summer/fall)
        - 'season_*' one-hot dummies (0/1)
        - 'is_tourist_season', 'is_weekend_tourist_season' (0/1)

    Notes
    -----
    Date range checks are string comparisons ('YYYY-MM'), not full timestamps.
    """
    df['covid_lockdown'] = ((df['date'] >= '2020-03') & (df['date'] <= '2020-06')).astype(int)
    df['covid_lockdown_lift'] = ((df['date'] >= '2020-06') & (df['date'] <= '2021-05')).astype(int)
    df['covid_lull'] = ((df['date'] >= '2021-06') & (df['date'] <= '2022-04')).astype(int)
    df['post_covid_recovery'] = ((df['date'] >= '2022-05') & (df['date'] <= '2022-12')).astype(int)

    df['season'] = pd.cut(df['month'], bins=[0,3,6,9,12], labels=['winter','spring','summer','fall'])
    df = pd.concat([df, pd.get_dummies(df['season'], prefix='season').astype(int)], axis=1)

    df['is_tourist_season'] = df['month'].isin([5,6,7,8,9,10]).astype(int)
    df['is_weekend_tourist_season'] = (df['is_weekend'] & df['is_tourist_season']).astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Select numeric feature columns for modeling.

    Excludes identifier/time/meta columns and raw categoricals that have been
    replaced by dummies.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    list of str
        Column names to use as features.

    Side Effects
    ------------
    Prints the set of removed columns for transparency.

    Notes
    -----
    Excludes:
    ['id','datetime','date','timestamp','streetname','city',
    'n_pedestrians','n_pedestrians_towards','n_pedestrians_away',
    'weather_condition','temp_band','season']
    """
    exclude_cols = ['id', 'datetime', 'date', 'weekday', 'timestamp', 'streetname', 'city',
                    'n_pedestrians', 'n_pedestrians_towards', 'n_pedestrians_away',
                    'weather_condition', 'temp_band', 'season']

    feature_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in exclude_cols]

    # Get the difference between df.columns and feature_cols and print all removed columns
    removed_cols = set(df.columns) - set(feature_cols)
    print(f'Removed columns: {removed_cols}')

    return feature_cols


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end feature builder for hourly pedestrian forecasting.

    Runs base/time/weather/seasonal/event features, timestamp and lag means,
    holiday enhancements, interactions, and street features. Finally removes
    selected categoricals and label-encodes remaining object columns (except
    identifiers).

    Parameters
    ----------
    df : pandas.DataFrame
        Minimum required columns:
        - 'date', 'hour', 'month', 'day_of_week', 'streetname'
        - 'weather_condition', 'temperature'
        - Any columns required by the merged CSVs (events/holidays)

    Returns
    -------
    pandas.DataFrame
        Feature-augmented frame ready for model input.

    Notes
    -----
    - Order matters only after timestamp creation; rolling features require 'timestamp'.
    - Drops raw 'city','weather_condition','temp_band','season' if present.
    - Label-encodes remaining object columns except ['id','streetname','date'].
    - Relies on external CSVs via GENERAL_DATA_PATH from config.
    """
    # The following functions don't need to be executed in a specific order
    df = create_base_time_features(df)
    df = create_time_block_features(df)
    df = create_weather_features(df)
    df = create_seasonal_features(df)
    df = add_wuerzburg_events(df)
    df = add_timestamp(df)

    # The following functions have to be executed after all previous ones
    # e.g., add_rolling_lag_means needs timestamp
    df = add_rolling_lag_means(df, value_cols=['temperature'])
    df = add_enhanced_holiday_features(df)
    df = create_interaction_features(df)
    df = add_street_features(df)

    #---

    ###### FURTHER PROCESSING ######

    # Remove categorical columns that have been replaced by dummies
    for col in ['city', 'weather_condition', 'temp_band', 'season']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encode categorical/object columns that are not id, streetname, date
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['id', 'streetname', 'date']:
            print(f'Encoding {col}')
            df[f'{col}_encoded'] = le.fit_transform(df[col])

    return df


def align_dataframe_columns(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           exclude_cols: list = None) -> pd.DataFrame:
    """
    Align test dataframe columns to match training dataframe.

    Adds missing columns (filled with 0) and removes extra columns from test_df
    to ensure both dataframes have the same feature columns.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataframe with complete set of columns
    test_df : pandas.DataFrame
        Test dataframe to align
    exclude_cols : list, optional
        Columns to exclude from alignment (e.g., target columns, ids)

    Returns
    -------
    pandas.DataFrame
        Test dataframe with aligned columns

    Notes
    -----
    This is useful when one-hot encoding creates different columns in train vs test
    due to missing categorical values.
    """
    if exclude_cols is None:
        exclude_cols = ['id', 'datetime', 'date', 'timestamp', 'streetname',
                       'n_pedestrians', 'n_pedestrians_towards', 'n_pedestrians_away']

    # Get column sets
    train_cols = set(train_df.columns) - set(exclude_cols)
    test_cols = set(test_df.columns) - set(exclude_cols)

    # Find missing and extra columns
    missing_cols = train_cols - test_cols
    extra_cols = test_cols - train_cols

    if missing_cols:
        print(f"Adding {len(missing_cols)} missing columns to test set: {sorted(missing_cols)}")
        for col in missing_cols:
            test_df[col] = 0

    if extra_cols:
        print(f"Removing {len(extra_cols)} extra columns from test set: {sorted(extra_cols)}")
        test_df = test_df.drop(columns=list(extra_cols))

    return test_df
