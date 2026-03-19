"""
preprocessor.py
---------------
Contains the CreditPreprocessor class which handles ALL data transformation:
  1. Feature engineering  (same logic as your original notebook)
  2. Missingness flags     (EXT_SOURCE_1/2/3 missing indicators)
  3. DAYS → years conversion + IS_UNEMPLOYED flag
  4. OCCUPATION_TYPE NaN → "Unknown" category
  5. LabelEncoding for categorical features
  6. StandardScaler for continuous numerical features
  7. Save / Load all fitted objects to disk (scaler.pkl + encoders.pkl + mappings JSONs)

Key design principle:
  fit_transform() is called ONCE on training data.
  transform()     is called on validation, test, and API inference data.
  The scaler and encoders are fitted only on training data to prevent
  data leakage.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# Columns that get StandardScaler applied
# Binary flags and already-bounded scores do NOT need scaling
COLS_TO_SCALE = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "LOAN_TO_GOODS_RATIO",
    "AGE_YEARS",
    "YEARS_EMPLOYED",
    "YEARS_ID_PUBLISH",
]

# These binary/indicator columns pass through unscaled
PASSTHROUGH_COLS = [
    "EXT_SOURCE_1_MISSING",
    "EXT_SOURCE_2_MISSING",
    "EXT_SOURCE_3_MISSING",
    "IS_UNEMPLOYED",
]


class CreditPreprocessor:
    """
    Stateful preprocessor for the credit scoring pipeline.

    Attributes
    ----------
    scaler       : fitted StandardScaler (for continuous numericals)
    encoders     : dict of {col_name: fitted LabelEncoder}
    cardinalities: dict of {col_name: n_unique_values} — needed by model_builder
    is_fitted    : bool
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders: dict[str, LabelEncoder] = {}
        self.cardinalities: dict[str, int] = {}
        self.is_fitted = False

    # ── Feature Engineering ───────────────────────────────────────────────────

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering to a raw cleaned DataFrame.
        Works identically on train and test data.

        Steps (all from your original notebook):
          1. Fill AMT_GOODS_PRICE NaN with AMT_CREDIT (0.99 correlation)
          2. Create LOAN_TO_GOODS_RATIO = AMT_CREDIT / AMT_GOODS_PRICE
          3. Drop AMT_GOODS_PRICE (redundant after ratio)
          4. Convert DAYS_BIRTH → AGE_YEARS (positive float)
          5. Handle DAYS_EMPLOYED: 365243 is a sentinel for unemployed
             → Create IS_UNEMPLOYED flag, replace sentinel with median
             → Convert to YEARS_EMPLOYED
          6. Convert DAYS_ID_PUBLISH → YEARS_ID_PUBLISH (positive float)
          7. Fill OCCUPATION_TYPE NaN → "Unknown" (preserves info)
          8. Create EXT_SOURCE missingness flags before imputation
          9. Impute EXT_SOURCE NaNs with median (computed per call — for
             train this is the training median; for test you must pass
             the training medians via the `ext_medians` argument)
        """
        df = df.copy()

        # 1–3: Loan-to-goods ratio
        if "AMT_GOODS_PRICE" in df.columns:
            df["AMT_GOODS_PRICE"] = df["AMT_GOODS_PRICE"].fillna(df["AMT_CREDIT"])
            df["LOAN_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
            df = df.drop(columns=["AMT_GOODS_PRICE"])

        # 4: Age in years (DAYS_BIRTH is negative)
        df["AGE_YEARS"] = df["DAYS_BIRTH"].abs() / 365.25
        df = df.drop(columns=["DAYS_BIRTH"])

        # 5: Employment
        UNEMPLOYED_SENTINEL = 365243
        df["IS_UNEMPLOYED"] = (df["DAYS_EMPLOYED"] == UNEMPLOYED_SENTINEL).astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(UNEMPLOYED_SENTINEL, np.nan)
        # Use median of non-sentinel values (will be overridden on test data)
        employed_median = df["DAYS_EMPLOYED"].median()
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].fillna(employed_median)
        df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].abs() / 365.25
        df = df.drop(columns=["DAYS_EMPLOYED"])

        # 6: ID publish age
        df["YEARS_ID_PUBLISH"] = df["DAYS_ID_PUBLISH"].abs() / 365.25
        df = df.drop(columns=["DAYS_ID_PUBLISH"])

        # 7: Fill missing occupation
        df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

        # 8: Missingness flags (BEFORE imputation — order matters)
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            df[f"{col}_MISSING"] = df[col].isna().astype(int)

        # 9: AMT_ANNUITY has small number of NaNs — fill with median
        if df["AMT_ANNUITY"].isna().sum() > 0:
            df["AMT_ANNUITY"] = df["AMT_ANNUITY"].fillna(df["AMT_ANNUITY"].median())

        return df

    # ── Fit + Transform (training data only) ─────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features, fit the scaler and encoders on training data,
        then return the fully processed DataFrame.

        Call this ONLY on training data.
        """
        df = self.engineer_features(df)

        # ── Categorical encoding ──
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
            self.cardinalities[col] = len(le.classes_)
            print(f"[preprocessor] {col}: {self.cardinalities[col]} unique categories")

        # ── EXT_SOURCE imputation with training medians ──
        # Store medians so we can apply the same values on test/API data
        self._ext_medians = {}
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            med = df[col].median()
            self._ext_medians[col] = med
            df[col] = df[col].fillna(med)

        # ── Scale continuous columns ──
        df[COLS_TO_SCALE] = self.scaler.fit_transform(df[COLS_TO_SCALE])

        self.is_fitted = True
        return df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["TARGET"]]

    # ── Transform only (test / API inference) ────────────────────────────────

    def transform(self, df: pd.DataFrame, has_target: bool = False) -> pd.DataFrame:
        """
        Apply the already-fitted scaler and encoders to new data.
        Does NOT refit anything — prevents data leakage.

        Parameters
        ----------
        df         : raw or cleaned DataFrame (same format as training input)
        has_target : set True if TARGET column is present (e.g. test set with labels)
        """
        if not self.is_fitted:
            raise RuntimeError("CreditPreprocessor is not fitted. Call fit_transform() first.")

        df = self.engineer_features(df)

        # ── Categorical encoding using fitted encoders ──
        for col in CATEGORICAL_FEATURES:
            le = self.encoders[col]
            # Handle unseen categories gracefully → map to index 0 ("Unknown")
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

        # ── EXT_SOURCE imputation with TRAINING medians ──
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            df[col] = df[col].fillna(self._ext_medians[col])

        # ── Scale using fitted scaler ──
        df[COLS_TO_SCALE] = self.scaler.transform(df[COLS_TO_SCALE])

        cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        if has_target:
            cols = cols + ["TARGET"]
        return df[cols]

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self,
             scaler_path: str = "models/scaler.pkl",
             encoders_path: str = "models/encoder_objects.pkl",
             mappings_dir: str = "data/mappings"):
        """
        Persist all fitted objects to disk.

        Saves:
          models/scaler.pkl          — fitted StandardScaler
          models/encoder_objects.pkl — dict of fitted LabelEncoders + medians + cardinalities
          data/mappings/<col>.json   — human-readable int→category name mapping per column
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save: preprocessor not yet fitted.")

        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        Path(encoders_path).parent.mkdir(parents=True, exist_ok=True)
        Path(mappings_dir).mkdir(parents=True, exist_ok=True)

        # Save scaler
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # Save encoders + metadata together
        payload = {
            "encoders": self.encoders,
            "cardinalities": self.cardinalities,
            "ext_medians": self._ext_medians,
        }
        with open(encoders_path, "wb") as f:
            pickle.dump(payload, f)

        # Save human-readable mappings (int → original label)
        for col, le in self.encoders.items():
            mapping = {int(i): str(label) for i, label in enumerate(le.classes_)}
            with open(f"{mappings_dir}/{col}.json", "w") as f:
                json.dump(mapping, f, indent=2)

        print(f"[preprocessor] Saved scaler → {scaler_path}")
        print(f"[preprocessor] Saved encoders → {encoders_path}")
        print(f"[preprocessor] Saved mappings → {mappings_dir}/")

    # ── Load ──────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls,
             scaler_path: str = "models/scaler.pkl",
             encoders_path: str = "models/encoder_objects.pkl"):
        """
        Reconstruct a fitted CreditPreprocessor from saved files.
        Use this in the API and in inference scripts.
        """
        obj = cls()

        with open(scaler_path, "rb") as f:
            obj.scaler = pickle.load(f)

        with open(encoders_path, "rb") as f:
            payload = pickle.load(f)

        obj.encoders = payload["encoders"]
        obj.cardinalities = payload["cardinalities"]
        obj._ext_medians = payload["ext_medians"]
        obj.is_fitted = True

        print(f"[preprocessor] Loaded scaler from {scaler_path}")
        print(f"[preprocessor] Loaded encoders from {encoders_path}")
        return obj

