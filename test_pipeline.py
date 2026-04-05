"""
test_pipeline.py
----------------
Unit tests for the Heart Disease Prediction pipeline.
Run with: pytest tests/
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_preprocessing import (
    clean_data,
    cap_outliers,
    engineer_features,
    preprocess,
    load_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic dataset mimicking the heart disease CSV."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age"     : np.random.randint(30, 75, n),
        "sex"     : np.random.randint(0, 2, n),
        "cp"      : np.random.randint(0, 4, n),
        "trestbps": np.random.randint(90, 180, n).astype(float),
        "chol"    : np.random.randint(150, 350, n).astype(float),
        "fbs"     : np.random.randint(0, 2, n),
        "restecg" : np.random.randint(0, 3, n),
        "thalach" : np.random.randint(80, 200, n).astype(float),
        "exang"   : np.random.randint(0, 2, n),
        "oldpeak" : np.random.uniform(0, 6, n).round(1),
        "slope"   : np.random.randint(0, 3, n),
        "ca"      : np.random.randint(0, 4, n),
        "thal"    : np.random.randint(1, 4, n),
        "target"  : np.random.randint(0, 2, n),
    })


@pytest.fixture
def dirty_df(sample_df):
    """Dataset with missing values and duplicates."""
    df = sample_df.copy()
    df.loc[0:4, "trestbps"] = np.nan
    df.loc[5:7, "ca"]       = np.nan
    df = pd.concat([df, df.iloc[:5]])  # add duplicates
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tests — clean_data
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanData:

    def test_removes_duplicates(self, dirty_df):
        original_len = len(dirty_df.drop_duplicates())
        cleaned = clean_data(dirty_df)
        assert len(cleaned) == original_len

    def test_no_missing_values_after_cleaning(self, dirty_df):
        cleaned = clean_data(dirty_df)
        assert cleaned.isnull().sum().sum() == 0

    def test_target_is_binary(self, dirty_df):
        cleaned = clean_data(dirty_df)
        assert set(cleaned["target"].unique()).issubset({0, 1})

    def test_returns_dataframe(self, sample_df):
        result = clean_data(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self, sample_df):
        original_shape = sample_df.shape
        clean_data(sample_df)
        assert sample_df.shape == original_shape


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tests — cap_outliers
# ─────────────────────────────────────────────────────────────────────────────

class TestCapOutliers:

    def test_outliers_are_capped(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "chol"] = 9999   # extreme outlier
        capped = cap_outliers(df, ["chol"])
        Q1 = sample_df["chol"].quantile(0.25)
        Q3 = sample_df["chol"].quantile(0.75)
        upper = Q3 + 1.5 * (Q3 - Q1)
        assert capped["chol"].max() <= upper + 1   # allow tiny float error

    def test_shape_unchanged(self, sample_df):
        result = cap_outliers(sample_df, ["trestbps", "chol"])
        assert result.shape == sample_df.shape

    def test_does_not_mutate_input(self, sample_df):
        before_max = sample_df["chol"].max()
        cap_outliers(sample_df, ["chol"])
        assert sample_df["chol"].max() == before_max


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tests — engineer_features
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineerFeatures:

    def test_new_columns_added(self, sample_df):
        result = engineer_features(sample_df)
        for col in ["age_group", "bp_category", "age_thalach", "chol_age_ratio"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nulls_in_new_features(self, sample_df):
        result = engineer_features(sample_df)
        for col in ["age_group", "bp_category", "age_thalach", "chol_age_ratio"]:
            assert result[col].isnull().sum() == 0

    def test_age_thalach_values(self, sample_df):
        result = engineer_features(sample_df)
        expected = (sample_df["age"] * sample_df["thalach"]).values
        np.testing.assert_array_equal(result["age_thalach"].values, expected)

    def test_age_group_range(self, sample_df):
        result = engineer_features(sample_df)
        assert result["age_group"].min() >= 0
        assert result["age_group"].max() <= 4

    def test_does_not_mutate_input(self, sample_df):
        original_cols = set(sample_df.columns)
        engineer_features(sample_df)
        assert set(sample_df.columns) == original_cols


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tests — preprocess
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocess:

    def test_output_shapes(self, sample_df):
        df = engineer_features(clean_data(sample_df))
        X_train, X_test, y_train, y_test, _ = preprocess(df, test_size=0.2)
        n = len(df)
        assert len(X_train) + len(X_test) == n
        assert len(y_train) + len(y_test) == n

    def test_stratified_split(self, sample_df):
        df = engineer_features(clean_data(sample_df))
        X_train, X_test, y_train, y_test, _ = preprocess(df, test_size=0.2, random_state=0)
        # Class ratios should be roughly equal
        ratio_train = y_train.mean()
        ratio_test  = y_test.mean()
        assert abs(ratio_train - ratio_test) < 0.15

    def test_scaler_returned(self, sample_df):
        df = engineer_features(clean_data(sample_df))
        _, _, _, _, scaler = preprocess(df, scale=True)
        assert scaler is not None

    def test_no_scaler_when_disabled(self, sample_df):
        df = engineer_features(clean_data(sample_df))
        _, _, _, _, scaler = preprocess(df, scale=False)
        assert scaler is None

    def test_feature_columns_preserved(self, sample_df):
        df = engineer_features(clean_data(sample_df))
        X_train, X_test, _, _, _ = preprocess(df)
        assert set(X_train.columns) == set(X_test.columns)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Integration Test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline(self, sample_df):
        """End-to-end: raw data → train/test split."""
        df_clean = clean_data(sample_df)
        df_capped = cap_outliers(df_clean, ["trestbps", "chol", "thalach", "oldpeak"])
        df_feat  = engineer_features(df_capped)
        X_train, X_test, y_train, y_test, scaler = preprocess(df_feat, test_size=0.2)

        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert scaler is not None

    def test_no_data_leakage(self, sample_df):
        """Train and test sets must not share indices."""
        df_feat = engineer_features(clean_data(sample_df))
        X_train, X_test, _, _, _ = preprocess(df_feat)
        shared = set(X_train.index) & set(X_test.index)
        assert len(shared) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
