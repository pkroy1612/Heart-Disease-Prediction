"""
data_preprocessing.py
---------------------
Handles all data loading, cleaning, feature engineering,
and train/test splitting for the Heart Disease Prediction project.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the heart disease CSV dataset.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Raw dataframe.
    """
    if not os.path.exists(filepath):
        # Auto-download from UCI if file is missing
        print("[INFO] Dataset not found locally. Downloading from UCI...")
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal", "target"
        ]
        df = pd.read_csv(url, names=cols, na_values="?")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"[INFO] Dataset saved to {filepath}")
    else:
        df = pd.read_csv(filepath)

    print(f"[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataframe:
    - Binarize target column
    - Handle missing values
    - Remove duplicates
    - Fix dtypes

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()

    # Binarize target: 0 = No Disease, 1 = Disease
    df["target"] = (df["target"] > 0).astype(int)

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Removed {dropped} duplicate rows.")

    # Handle missing values
    missing = df.isnull().sum()
    cols_with_na = missing[missing > 0].index.tolist()
    if cols_with_na:
        print(f"[INFO] Missing values found in: {cols_with_na}")
        for col in cols_with_na:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        print("[INFO] Missing values imputed with median/mode.")

    # Ensure correct dtypes for categorical columns
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)
            df[col] = df[col].astype(int)

    print(f"[INFO] Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 3. OUTLIER HANDLING
# ─────────────────────────────────────────────

def cap_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Cap outliers using IQR method (Winsorization).

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list): Columns to apply IQR capping.

    Returns:
        pd.DataFrame: Dataframe with capped outliers.
    """
    df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if before:
            print(f"[INFO] Capped {before} outliers in '{col}'.")
    return df


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing clinical data.

    New features:
    - age_group       : Binned age categories
    - bp_category     : Blood pressure risk category
    - age_thalach     : Interaction: age × max heart rate
    - chol_age_ratio  : Cholesterol-to-age ratio

    Args:
        df (pd.DataFrame): Cleaned dataframe.

    Returns:
        pd.DataFrame: Dataframe with new features.
    """
    df = df.copy()

    # Age group bins
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 50, 60, 70, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # Blood pressure category
    df["bp_category"] = pd.cut(
        df["trestbps"],
        bins=[0, 120, 129, 139, 180, 300],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # Interaction: age × max heart rate (higher in younger patients = better)
    df["age_thalach"] = df["age"] * df["thalach"]

    # Cholesterol to age ratio
    df["chol_age_ratio"] = (df["chol"] / df["age"]).round(2)

    print(f"[INFO] Feature engineering done. New shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 5. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True
):
    """
    Full preprocessing pipeline:
    - Split features / target
    - Train/test split
    - Optional StandardScaler

    Args:
        df (pd.DataFrame): Engineered dataframe.
        test_size (float): Fraction for test split.
        random_state (int): Random seed.
        scale (bool): Whether to apply StandardScaler.

    Returns:
        X_train, X_test, y_train, y_test, scaler (or None)
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        print("[INFO] Features scaled with StandardScaler.")

    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler


# ─────────────────────────────────────────────
# 6. FULL PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_preprocessing(filepath: str = "data/heart.csv"):
    """
    Orchestrates the full preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test, scaler, df_clean
    """
    df_raw    = load_data(filepath)
    df_clean  = clean_data(df_raw)

    # Cap outliers in continuous columns
    continuous_cols = ["trestbps", "chol", "thalach", "oldpeak"]
    df_clean  = cap_outliers(df_clean, continuous_cols)

    df_feat   = engineer_features(df_clean)

    # Save processed data
    out_path = filepath.replace("heart.csv", "heart_processed.csv")
    df_feat.to_csv(out_path, index=False)
    print(f"[INFO] Processed data saved to: {out_path}")

    X_train, X_test, y_train, y_test, scaler = preprocess(df_feat)
    return X_train, X_test, y_train, y_test, scaler, df_feat


if __name__ == "__main__":
    run_preprocessing()
