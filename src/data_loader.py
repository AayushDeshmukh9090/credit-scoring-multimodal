"""
data_loader.py
--------------
Responsible for:
  1. Defining which columns are numerical vs categorical (single source of truth)
  2. Loading raw CSVs from data/raw/
  3. Splitting processed data into train/validation sets

Why this exists as a separate module:
  The notebook, the training script, AND the API all need to know the same
  column names. If you hard-code them in three places, they will drift.
  Define once here, import everywhere.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
# from pathlib import Path
# ── Column Definitions (Single Source of Truth) ──────────────────────────────
# These are the FINAL feature columns after cleaning and engineering.
# Any change to features must be made here and only here.

NUMERICAL_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "LOAN_TO_GOODS_RATIO",
    "EXT_SOURCE_1_MISSING",   # Missingness indicator flag
    "EXT_SOURCE_2_MISSING",
    "EXT_SOURCE_3_MISSING",
    "AGE_YEARS",
    "YEARS_EMPLOYED",
    "YEARS_ID_PUBLISH",
    "IS_UNEMPLOYED",          # Engineered flag: DAYS_EMPLOYED == 365243
]

CATEGORICAL_FEATURES = [
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "CODE_GENDER",
]

TARGET_COL = "TARGET"

# Sensitive attributes used for fairness evaluation (not model inputs)
SENSITIVE_FEATURES = ["CODE_GENDER"]

# ROOT = Path(__file__).resolve().parent.parent

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_raw_train(path: str = "../data/raw/application_train.csv") -> pd.DataFrame:
    """Load the original unmodified training CSV."""
    df = pd.read_csv(path)
    print(f"[data_loader] Loaded train: {df.shape}")
    return df


def load_raw_test(path: str = "../data/raw/application_test.csv") -> pd.DataFrame:
    """Load the original unmodified test CSV."""
    df = pd.read_csv(path)
    print(f"[data_loader] Loaded test: {df.shape}")
    return df


def load_processed_train(path: str = "../data/processed/train_processed.parquet") -> pd.DataFrame:
    """Load the cleaned + preprocessed training data."""
    df = pd.read_parquet(path)
    print(f"[data_loader] Loaded processed train: {df.shape}")
    return df


def load_processed_test(path: str = "../data/processed/test_processed.parquet") -> pd.DataFrame:
    """Load the cleaned + preprocessed test data."""
    df = pd.read_parquet(path)
    print(f"[data_loader] Loaded processed test: {df.shape}")
    return df


# ── Splitter ──────────────────────────────────────────────────────────────────

def split_features_and_target(df: pd.DataFrame):
    """
    Separate a processed DataFrame into:
      - X_num : numerical features only
      - X_cat : categorical features only (integer-encoded)
      - y     : target series

    Returns
    -------
    X_num : pd.DataFrame
    X_cat : pd.DataFrame
    y     : pd.Series
    """
    X_num = df[NUMERICAL_FEATURES]
    X_cat = df[CATEGORICAL_FEATURES]
    y = df[TARGET_COL]
    return X_num, X_cat, y


def train_val_split(X_num, X_cat, y, val_size: float = 0.2, random_state: int = 42):
    """
    Splits into train and validation sets.
    Stratified on y to preserve class imbalance ratio in both splits.

    Returns
    -------
    X_num_tr, X_num_val, X_cat_tr, X_cat_val, y_tr, y_val
    """
    idx = X_num.index
    idx_tr, idx_val = train_test_split(
        idx, test_size=val_size, random_state=random_state, stratify=y
    )
    return (
        X_num.loc[idx_tr],
        X_num.loc[idx_val],
        X_cat.loc[idx_tr],
        X_cat.loc[idx_val],
        y.loc[idx_tr],
        y.loc[idx_val],
    )


def compute_class_weights(y: pd.Series) -> dict:
    """
    Compute balanced class weights for the imbalanced target.
    Formula: weight_c = total_count / (2 * count_c)

    This is the same formula used in the original notebook.
    Centralised here so all training scripts use identical logic.
    """
    total = len(y)
    count_0 = (y == 0).sum()
    count_1 = (y == 1).sum()
    w0 = total / (2 * count_0)
    w1 = total / (2 * count_1)
    print(f"[data_loader] Class weights → 0: {w0:.4f}, 1: {w1:.4f}")
    return {0: w0, 1: w1}

