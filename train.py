"""
train.py
--------
Model training, hyperparameter tuning, and cross-validation
for the Heart Disease Prediction project.

Models: KNN, SVM (RBF), Logistic Regression, Random Forest
Tuning : GridSearchCV with 5-fold Stratified CV
"""

import os
import sys
import time
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.neighbors        import KNeighborsClassifier
from sklearn.svm              import SVC
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_preprocessing import run_preprocessing

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Definitions & Hyperparameter Grids
# ─────────────────────────────────────────────────────────────────────────────

def get_models_and_grids() -> dict:
    """
    Returns a dict of:
        model_name -> {"model": estimator, "params": param_grid}
    """
    return {
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
                "weights"     : ["uniform", "distance"],
                "metric"      : ["euclidean", "manhattan", "minkowski"],
                "p"           : [1, 2],
            }
        },
        "SVM_RBF": {
            "model": SVC(kernel="rbf", probability=True, random_state=42),
            "params": {
                "C"    : [0.01, 0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            }
        },
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C"      : [0.01, 0.1, 1, 10, 100],
                "solver" : ["lbfgs", "liblinear"],
                "penalty": ["l2"],
            }
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators"    : [50, 100, 200],
                "max_depth"       : [None, 5, 10, 15],
                "min_samples_split": [2, 5, 10],
                "max_features"    : ["sqrt", "log2"],
            }
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hyperparameter Tuning (GridSearchCV)
# ─────────────────────────────────────────────────────────────────────────────

def tune_model(
    name: str,
    model,
    param_grid: dict,
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "f1"
) -> GridSearchCV:
    """
    Run GridSearchCV for a single model.

    Args:
        name       : Model display name.
        model      : Sklearn estimator.
        param_grid : Hyperparameter grid.
        X_train    : Training features.
        y_train    : Training labels.
        cv         : Number of folds.
        scoring    : Optimization metric.

    Returns:
        Fitted GridSearchCV object.
    """
    print(f"\n{'─'*55}")
    print(f"  Tuning: {name}")
    print(f"{'─'*55}")

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    t0 = time.time()
    grid = GridSearchCV(
        estimator  = model,
        param_grid = param_grid,
        cv         = cv_strategy,
        scoring    = scoring,
        refit      = True,
        n_jobs     = -1,
        verbose    = 0
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Best Params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")
    print(f"  Time Taken  : {elapsed:.1f}s")
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluate Model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """
    Compute classification metrics on the test set.

    Returns dict with accuracy, precision, recall, f1, roc_auc.
    """
    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, "predict_proba") else None)

    metrics = {
        "model"    : name,
        "accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall"   : round(recall_score(y_test, y_pred), 4),
        "f1"       : round(f1_score(y_test, y_pred), 4),
        "roc_auc"  : round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-Validation Score
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(name: str, model, X, y, cv: int = 5) -> dict:
    """
    Perform cross-validation on the full dataset (train + test).

    Returns mean ± std for accuracy and F1.
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    acc_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring="accuracy")
    f1_scores  = cross_val_score(model, X, y, cv=cv_strategy, scoring="f1")

    return {
        "model"          : name,
        "cv_accuracy_mean": round(acc_scores.mean(), 4),
        "cv_accuracy_std" : round(acc_scores.std(), 4),
        "cv_f1_mean"      : round(f1_scores.mean(), 4),
        "cv_f1_std"       : round(f1_scores.std(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Save Model
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, name: str):
    """Persist a trained model to disk using joblib."""
    path = os.path.join(MODELS_DIR, f"{name.lower()}_model.pkl")
    joblib.dump(model, path)
    print(f"  [SAVED] Model → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Print Results Table
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list):
    """Print a formatted results comparison table."""
    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False)
    print("\n" + "="*75)
    print("  MODEL PERFORMANCE COMPARISON (Test Set)")
    print("="*75)
    print(df.to_string(index=False))
    print("="*75 + "\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_training(filepath: str = "data/heart.csv"):
    """
    Orchestrate the full training pipeline:
      1. Preprocess data
      2. Tune each model
      3. Evaluate on test set
      4. Cross-validate
      5. Save best models
      6. Print results
    """
    print("\n" + "="*55)
    print("  HEART DISEASE PREDICTION — TRAINING PIPELINE")
    print("="*55)

    # ── Data ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, df = run_preprocessing(filepath)
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    # ── Models ────────────────────────────────────────────
    model_configs = get_models_and_grids()
    test_results  = []
    cv_results    = []
    trained_models = {}

    for name, config in model_configs.items():
        # Tune
        grid = tune_model(
            name, config["model"], config["params"],
            X_train, y_train, cv=5, scoring="f1"
        )
        best_model = grid.best_estimator_

        # Test set evaluation
        test_metrics = evaluate_model(name, best_model, X_test, y_test)
        test_results.append(test_metrics)

        # CV evaluation
        cv_metrics = cross_validate_model(name, best_model, X_all, y_all, cv=5)
        cv_results.append(cv_metrics)

        # Save
        save_model(best_model, name)
        trained_models[name] = best_model

        # Classification report
        y_pred = best_model.predict(X_test)
        print(f"\n  Classification Report — {name}:")
        print(classification_report(y_test, y_pred,
              target_names=["No Disease", "Disease"]))

    # ── Results ───────────────────────────────────────────
    results_df = print_results_table(test_results)

    print("\nCross-Validation Results (Mean ± Std, 5-Fold):")
    print(pd.DataFrame(cv_results).to_string(index=False))

    # ── Save scaler ───────────────────────────────────────
    if scaler is not None:
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"\n[SAVED] Scaler → {scaler_path}")

    # ── Best model summary ────────────────────────────────
    best_row = results_df.iloc[0]
    print(f"\n✅ Best Model : {best_row['model']}")
    print(f"   Accuracy   : {best_row['accuracy']*100:.1f}%")
    print(f"   Precision  : {best_row['precision']*100:.1f}%")
    print(f"   Recall     : {best_row['recall']*100:.1f}%")
    print(f"   F1-Score   : {best_row['f1']*100:.1f}%")

    return trained_models, results_df


if __name__ == "__main__":
    run_training()
