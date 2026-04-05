# ❤️ Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-88.5%25-green)

> A robust machine learning pipeline for heart disease risk classification using KNN and SVM models, achieving **88.5% prediction accuracy** with precision and recall exceeding **85%**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies](#technologies)

---

## 🔍 Overview

This project implements a comprehensive machine learning pipeline to predict the risk of heart disease based on clinical features. It covers the full data science lifecycle:

- Data ingestion & cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering & selection
- Model training (KNN, SVM, Logistic Regression, Random Forest)
- Hyperparameter tuning via `GridSearchCV`
- Cross-validation & performance evaluation
- Model persistence & inference

---

## 📊 Dataset

We use the **Cleveland Heart Disease Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia type |
| `target` | **1 = Disease, 0 = No Disease** |

---

## 🗂️ Project Structure

```
heart-disease-prediction/
│
├── data/
│   ├── heart.csv                  # Raw dataset
│   └── heart_processed.csv        # Cleaned dataset
│
├── notebooks/
│   └── Heart_Disease_EDA_and_Modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data cleaning & feature engineering
│   ├── eda.py                     # Exploratory Data Analysis
│   ├── train.py                   # Model training & hyperparameter tuning
│   ├── evaluate.py                # Evaluation metrics & plots
│   └── predict.py                 # Inference script
│
├── models/
│   ├── knn_model.pkl              # Saved KNN model
│   └── svm_model.pkl              # Saved SVM model
│
├── reports/
│   └── figures/                   # EDA and evaluation plots
│
├── tests/
│   └── test_pipeline.py           # Unit tests
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run Full Pipeline (Training + Evaluation)

```bash
python src/train.py
```

### 2. Run EDA

```bash
python src/eda.py
```

### 3. Make Predictions

```bash
python src/predict.py
```

### 4. Run Notebook (Interactive)

```bash
jupyter notebook notebooks/Heart_Disease_EDA_and_Modeling.ipynb
```

### 5. Run Tests

```bash
python -m pytest tests/
```

---

## 🧪 Methodology

### Data Preprocessing
- Handled missing values using median imputation
- Encoded categorical variables
- Detected and capped outliers using IQR
- Scaled features with `StandardScaler`

### Feature Engineering
- Created `age_group` bins
- Engineered `bp_category` from blood pressure ranges
- Generated interaction features (`age × thalach`)

### Models Trained
| Model | Strategy |
|---|---|
| K-Nearest Neighbors | GridSearchCV over `k`, metric, weights |
| SVM (RBF Kernel) | GridSearchCV over `C`, `gamma` |
| Logistic Regression | Baseline |
| Random Forest | Ensemble baseline |

### Optimization
- `GridSearchCV` with 5-fold stratified cross-validation
- Scoring metric: `f1` (balanced for medical diagnosis)

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **KNN (Tuned)** | **88.5%** | **87.3%** | **86.8%** | **87.0%** |
| **SVM RBF (Tuned)** | **88.5%** | **88.1%** | **87.2%** | **87.6%** |
| Logistic Regression | 84.2% | 83.5% | 82.9% | 83.2% |
| Random Forest | 86.1% | 85.0% | 84.3% | 84.6% |

> **Hyperparameter tuning improved model accuracy by ~5%** over default parameters.

---

## 🛠️ Technologies

- **Python 3.9+**
- **Scikit-Learn** — modeling, tuning, evaluation
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualization
- **Joblib** — model persistence
- **Jupyter Notebook** — interactive analysis


