"""
Configuration for the Foot Traffic Forecasting Pipeline.

This file contains all configurable settings for the pipeline.
Your colleague only needs to edit BASE_PATH to run on their machine.
"""
from pathlib import Path

# ============================================================
# PATH CONFIGURATION - EDIT THIS FOR YOUR MACHINE
# ============================================================
BASE_PATH = Path("./")

# Derived paths (no changes needed below this line)
DATA_PATH = BASE_PATH / "data_foot_traffic"
GENERAL_DATA_PATH = BASE_PATH / "data_general"
OUTPUT_PATH = DATA_PATH / "submission.csv"

# ============================================================
# MODEL CONFIGURATION
# ============================================================
# Which models to train. Options: 'lgb' (LightGBM), 'xgb' (XGBoost)
MODELS_TO_USE = ['xgb']

# Hyperparameter tuning settings
TUNE_MODELS = False       # If True, run Bayesian hyperparameter tuning
N_SPLITS = 10             # Number of CV splits for tuning
N_ITER = 100              # Number of iterations for tuning
N_POINTS_BAYES = 5        # Points per iteration (Bayesian search)
N_JOBS = 5                # Parallel jobs for tuning
VERBOSE = 1               # Verbosity level for tuning output

# Training options
RETRAIN_ON_VAL = False    # If True, retrain on train+val for final predictions
DESEASONALIZE = False     # If True, apply deseasonalization to targets

# Train/Validation/Test split configuration
TRAIN_END_DATE = '2025-01-19'  # Last date included in training (validation starts next day)
VAL_TEST_DAYS = 14              # Number of days for validation and test periods (kept equal)

# Target columns to predict
TARGETS = ['n_pedestrians', 'n_pedestrians_towards', 'n_pedestrians_away']

# ============================================================
# MODEL HYPERPARAMETERS (used when TUNE_MODELS = False)
# ============================================================
LGB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'max_depth': 8,
    'num_leaves': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 0.1,
    'n_jobs': 1,
    'random_state': 42,
    'verbose': -1,
}

XGB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.02,
    'max_depth': 7,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': 1,
    'random_state': 42,
}
