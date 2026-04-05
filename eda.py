"""
eda.py
------
Exploratory Data Analysis for the Heart Disease Prediction project.
Generates and saves all EDA plots to reports/figures/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────────
PALETTE   = ["#E74C3C", "#2ECC71"]   # Red = Disease, Green = No Disease
SAVE_DIR  = "reports/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


def _save(fig: plt.Figure, name: str):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    """Print dataset summary statistics."""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Shape   : {df.shape}")
    print(f"Target  : {df['target'].value_counts().to_dict()}")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nData Types:\n{df.dtypes}")
    print("\nDescriptive Statistics:")
    print(df.describe().round(2))
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Target Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame):
    """Bar chart of class balance."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts()
    bars = ax.bar(
        ["No Disease", "Disease"],
        counts.values,
        color=PALETTE[::-1],
        edgecolor="white",
        width=0.5
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts.values) * 1.15)
    _save(fig, "target_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame):
    """Histograms with KDE for all numeric features, split by target."""
    num_cols = df.select_dtypes(include=np.number).columns.drop("target").tolist()
    n = len(num_cols)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        for label, color in zip([0, 1], PALETTE[::-1]):
            subset = df[df["target"] == label][col]
            axes[i].hist(subset, bins=20, alpha=0.6, color=color,
                         label=("No Disease" if label == 0 else "Disease"),
                         edgecolor="white", density=True)
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles = [
        mpatches.Patch(color=PALETTE[1], label="No Disease"),
        mpatches.Patch(color=PALETTE[0], label="Disease"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10)
    fig.suptitle("Feature Distributions by Target", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "feature_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    """Pearson correlation heatmap."""
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.5,
        ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Top Feature Correlations with Target
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_target_correlation(df: pd.DataFrame):
    """Horizontal bar chart of feature correlations with target."""
    corr = df.corr(numeric_only=True)["target"].drop("target").sort_values()
    colors = ["#E74C3C" if v < 0 else "#2ECC71" for v in corr.values]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(corr.index, corr.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pearson Correlation")
    ax.set_title("Feature Correlation with Target", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "feature_target_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Boxplots
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame):
    """Boxplots of continuous features by target."""
    continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, len(continuous), figsize=(18, 5))

    for ax, col in zip(axes, continuous):
        data_0 = df[df["target"] == 0][col]
        data_1 = df[df["target"] == 1][col]
        bp = ax.boxplot(
            [data_0, data_1],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2)
        )
        bp["boxes"][0].set_facecolor(PALETTE[1])
        bp["boxes"][1].set_facecolor(PALETTE[0])
        ax.set_xticklabels(["No Disease", "Disease"])
        ax.set_title(col, fontweight="bold")

    fig.suptitle("Continuous Features by Target Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "boxplots.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Categorical Feature Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_categorical_features(df: pd.DataFrame):
    """Grouped bar charts for categorical features vs. target."""
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ct = df.groupby([col, "target"]).size().unstack(fill_value=0)
        ct.plot(kind="bar", ax=axes[i], color=PALETTE[::-1], edgecolor="white",
                rot=0, legend=(i == 0))
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")
        if i == 0:
            axes[i].legend(["No Disease", "Disease"])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Features vs Target", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "categorical_features.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pairplot (Key Features)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pairplot(df: pd.DataFrame):
    """Pairplot of key clinical features."""
    key = ["age", "thalach", "oldpeak", "chol", "target"]
    df_sub = df[key].copy()
    df_sub["target"] = df_sub["target"].map({0: "No Disease", 1: "Disease"})

    g = sns.pairplot(
        df_sub, hue="target",
        palette={"No Disease": PALETTE[1], "Disease": PALETTE[0]},
        diag_kind="kde", plot_kws={"alpha": 0.5}
    )
    g.fig.suptitle("Pairplot of Key Clinical Features", y=1.02,
                   fontsize=14, fontweight="bold")
    _save(g.fig, "pairplot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Age Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_age_analysis(df: pd.DataFrame):
    """Age distribution + heart disease rate by decade."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Age distribution by target
    for label, color in zip([0, 1], PALETTE[::-1]):
        ax1.hist(df[df["target"] == label]["age"], bins=20, alpha=0.7,
                 color=color, label="No Disease" if label == 0 else "Disease",
                 edgecolor="white")
    ax1.set_title("Age Distribution by Target", fontweight="bold")
    ax1.set_xlabel("Age")
    ax1.legend()

    # Disease rate by decade
    df["decade"] = (df["age"] // 10 * 10).astype(str) + "s"
    rate = df.groupby("decade")["target"].mean() * 100
    ax2.bar(rate.index, rate.values, color="#E74C3C", edgecolor="white")
    ax2.set_title("Heart Disease Rate by Age Decade (%)", fontweight="bold")
    ax2.set_xlabel("Age Group")
    ax2.set_ylabel("Disease Rate (%)")

    df.drop(columns=["decade"], inplace=True)
    plt.tight_layout()
    _save(fig, "age_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL EDA
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame):
    """Run the complete EDA pipeline."""
    print("\n[EDA] Starting Exploratory Data Analysis...")
    print_summary(df)
    plot_target_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_feature_target_correlation(df)
    plot_boxplots(df)
    plot_categorical_features(df)
    plot_age_analysis(df)
    try:
        plot_pairplot(df)
    except Exception as e:
        print(f"[WARN] Pairplot skipped: {e}")
    print(f"\n[EDA] All plots saved to: {SAVE_DIR}/\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_preprocessing import run_preprocessing
    _, _, _, _, _, df = run_preprocessing()
    run_eda(df)
