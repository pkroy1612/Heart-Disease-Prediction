"""
predict.py
----------
Inference script for Heart Disease Prediction.
Loads saved models and makes predictions on new clinical data.

Usage:
    python src/predict.py
    python src/predict.py --model svm_rbf
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = "models"

# ── Feature order used during training ───────────────────────────────────────
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal",
    # Engineered features
    "age_group", "bp_category", "age_thalach", "chol_age_ratio"
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str = "svm_rbf"):
    """
    Load a saved model from disk.

    Args:
        model_name (str): One of 'knn', 'svm_rbf', 'logistic_regression',
                          'random_forest'.

    Returns:
        Fitted sklearn estimator.
    """
    path = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found at '{path}'.\n"
            f"Run `python src/train.py` first to train and save models."
        )
    model = joblib.load(path)
    print(f"[INFO] Loaded model: {path}")
    return model


def load_scaler():
    """Load the fitted StandardScaler."""
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Scaler not found. Run `python src/train.py` first."
        )
    scaler = joblib.load(path)
    print(f"[INFO] Loaded scaler: {path}")
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering (for new input)
# ─────────────────────────────────────────────────────────────────────────────

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering used during training.

    Args:
        df: DataFrame with raw clinical features.

    Returns:
        DataFrame with engineered features added.
    """
    df = df.copy()

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 50, 60, 70, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    df["bp_category"] = pd.cut(
        df["trestbps"],
        bins=[0, 120, 129, 139, 180, 300],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    df["age_thalach"]    = df["age"] * df["thalach"]
    df["chol_age_ratio"] = (df["chol"] / df["age"]).round(2)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(patient_data: dict, model_name: str = "svm_rbf") -> dict:
    """
    Predict heart disease risk for a single patient.

    Args:
        patient_data (dict): Raw clinical feature values.
        model_name   (str) : Which model to use.

    Returns:
        dict with prediction, probability, and risk level.
    """
    # Load artifacts
    model  = load_model(model_name)
    scaler = load_scaler()

    # Build dataframe
    df = pd.DataFrame([patient_data])

    # Feature engineering
    df = apply_feature_engineering(df)

    # Ensure all features present
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing feature: '{col}'")

    df = df[FEATURE_COLS]

    # Scale
    X = scaler.transform(df)

    # Predict
    prediction = model.predict(X)[0]
    probability = (
        model.predict_proba(X)[0][1]
        if hasattr(model, "predict_proba")
        else None
    )

    # Risk level
    if probability is not None:
        if probability < 0.30:
            risk = "Low"
        elif probability < 0.60:
            risk = "Moderate"
        else:
            risk = "High"
    else:
        risk = "High" if prediction == 1 else "Low"

    return {
        "prediction"  : int(prediction),
        "diagnosis"   : "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "probability" : round(float(probability), 4) if probability else None,
        "risk_level"  : risk,
        "model_used"  : model_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Batch Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(filepath: str, model_name: str = "svm_rbf") -> pd.DataFrame:
    """
    Make predictions on a CSV file of patients.

    Args:
        filepath   (str): Path to CSV with clinical features.
        model_name (str): Model to use.

    Returns:
        DataFrame with predictions appended.
    """
    model  = load_model(model_name)
    scaler = load_scaler()

    df = pd.read_csv(filepath)
    df_feat = apply_feature_engineering(df)
    df_feat = df_feat[FEATURE_COLS]

    X = scaler.transform(df_feat)
    predictions  = model.predict(X)
    probabilities = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else [None] * len(X)
    )

    df["prediction"]   = predictions
    df["probability"]  = [round(p, 4) if p else None for p in probabilities]
    df["diagnosis"]    = df["prediction"].map(
        {0: "No Heart Disease", 1: "Heart Disease Detected"}
    )

    out_path = filepath.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Batch predictions saved to: {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEMO
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_PATIENTS = [
    {
        "label"   : "Patient A — Low Risk (Expected: No Disease)",
        "data"    : {
            "age": 45, "sex": 1, "cp": 0, "trestbps": 118,
            "chol": 210, "fbs": 0, "restecg": 0, "thalach": 165,
            "exang": 0, "oldpeak": 0.5, "slope": 2, "ca": 0, "thal": 2
        }
    },
    {
        "label"   : "Patient B — High Risk (Expected: Disease)",
        "data"    : {
            "age": 62, "sex": 1, "cp": 2, "trestbps": 150,
            "chol": 268, "fbs": 0, "restecg": 1, "thalach": 120,
            "exang": 1, "oldpeak": 3.5, "slope": 0, "ca": 2, "thal": 3
        }
    },
    {
        "label"   : "Patient C — Moderate Risk (Female, 55)",
        "data"    : {
            "age": 55, "sex": 0, "cp": 1, "trestbps": 132,
            "chol": 240, "fbs": 0, "restecg": 0, "thalach": 142,
            "exang": 0, "oldpeak": 1.8, "slope": 1, "ca": 0, "thal": 2
        }
    },
]


def run_demo(model_name: str = "svm_rbf"):
    """Run predictions on sample patients and display results."""
    print("\n" + "="*60)
    print("  HEART DISEASE PREDICTION — DEMO INFERENCE")
    print("="*60)

    for patient in SAMPLE_PATIENTS:
        print(f"\n📋 {patient['label']}")
        print(f"   Input: {patient['data']}")
        try:
            result = predict(patient["data"], model_name=model_name)
            print(f"   🔍 Diagnosis  : {result['diagnosis']}")
            print(f"   📊 Probability: {result['probability']:.2%}" if result['probability'] else "")
            print(f"   ⚠️  Risk Level : {result['risk_level']}")
        except FileNotFoundError as e:
            print(f"   ❌ {e}")

    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease Predictor")
    parser.add_argument(
        "--model", type=str, default="svm_rbf",
        choices=["knn", "svm_rbf", "logistic_regression", "random_forest"],
        help="Model to use for inference"
    )
    parser.add_argument(
        "--batch", type=str, default=None,
        help="Path to CSV for batch prediction"
    )
    args = parser.parse_args()

    if args.batch:
        predict_batch(args.batch, model_name=args.model)
    else:
        run_demo(model_name=args.model)
