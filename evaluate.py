"""
evaluate.py
-----------
Model evaluation utilities: confusion matrices, ROC curves,
learning curves, and feature importance plots.
"""

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import learning_curve, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

SAVE_DIR = "reports/figures"
os.makedirs(SAVE_DIR, exist_ok=True)
PALETTE  = ["#2ECC71", "#E74C3C"]


def _save(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(models: dict, X_test, y_test):
    """
    Plot confusion matrices for all models side by side.

    Args:
        models (dict): {name: fitted_estimator}
        X_test, y_test: Test data.
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Disease", "Disease"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name.replace("_", " "), fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "confusion_matrices.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROC Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(models: dict, X_test, y_test):
    """
    Plot ROC curves for all models on a single axes.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name.replace('_', ' ')} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, "roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_recall_curves(models: dict, X_test, y_test):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

    for (name, model), color in zip(models.items(), colors):
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{name.replace('_', ' ')} (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, "precision_recall_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Learning Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(models: dict, X_train, y_train):
    """
    Plot learning curves (train vs. CV score vs. training size) for each model.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 10)

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=cv, scoring="f1", n_jobs=-1
        )
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        ax.plot(sizes, train_mean, "o-", color="#E74C3C", label="Train")
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="#E74C3C")
        ax.plot(sizes, val_mean, "o-", color="#2ECC71", label="Validation")
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color="#2ECC71")
        ax.set_title(name.replace("_", " "), fontweight="bold")
        ax.set_xlabel("Training Samples")
        ax.set_ylabel("F1 Score")
        ax.legend()
        ax.set_ylim(0.5, 1.05)

    fig.suptitle("Learning Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "learning_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature Importance (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, top_n: int = 15):
    """
    Plot feature importances for tree-based models.

    Args:
        model        : Fitted RandomForestClassifier.
        feature_names: List of feature names.
        top_n        : Show top N features.
    """
    if not hasattr(model, "feature_importances_"):
        print("[WARN] Model has no feature_importances_ attribute. Skipping.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [
        f"#{int(255 * (1 - v/max(top_values))):02x}"
        f"{int(255 * v/max(top_values)):02x}50"
        for v in top_values
    ]
    bars = ax.barh(top_features[::-1], top_values[::-1], color="#E74C3C",
                   edgecolor="white", alpha=0.85)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results_df):
    """
    Bar chart comparing accuracy, precision, recall, F1 across models.

    Args:
        results_df: DataFrame with columns [model, accuracy, precision, recall, f1].
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#3498DB", "#2ECC71", "#F39C12", "#E74C3C"]
    n_models  = len(results_df)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = results_df[metric].values
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=color, edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in results_df["model"].tolist()],
        fontsize=10
    )
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.axhline(0.85, color="black", linestyle="--", linewidth=0.8, alpha=0.4,
               label="85% threshold")
    plt.tight_layout()
    _save(fig, "model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. RUN ALL EVALUATIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(models: dict, X_train, X_test, y_train, y_test,
                   results_df=None, feature_names=None):
    """Run the full evaluation pipeline."""
    print("\n[EVAL] Generating evaluation plots...")
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_precision_recall_curves(models, X_test, y_test)
    plot_learning_curves(models, X_train, y_train)

    if "Random_Forest" in models:
        fn = feature_names or list(range(X_test.shape[1]))
        plot_feature_importance(models["Random_Forest"], fn)

    if results_df is not None:
        plot_model_comparison(results_df)

    print(f"[EVAL] All evaluation plots saved to: {SAVE_DIR}/\n")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_preprocessing import run_preprocessing
    from src.train import run_training

    X_train, X_test, y_train, y_test, scaler, df = run_preprocessing()
    trained_models, results_df = run_training()

    run_evaluation(
        trained_models, X_train, X_test, y_train, y_test,
        results_df=results_df, feature_names=list(X_test.columns)
    )
