# Data Directory

## heart.csv / heart_processed.csv

This folder stores the Cleveland Heart Disease dataset from the UCI ML Repository.

**The dataset is auto-downloaded** the first time you run:

```bash
python src/data_preprocessing.py
# or
python src/train.py
```

### Source
- **Name**: Cleveland Heart Disease Dataset  
- **URL**: https://archive.ics.uci.edu/ml/datasets/Heart+Disease  
- **Records**: 303 patients  
- **Features**: 13 clinical features + 1 target

### Feature Dictionary

| Column    | Type        | Description                                      |
|-----------|-------------|--------------------------------------------------|
| age       | Continuous  | Age in years                                     |
| sex       | Binary      | 1 = male, 0 = female                             |
| cp        | Categorical | Chest pain type (0=typical angina, …, 3=asymptomatic) |
| trestbps  | Continuous  | Resting blood pressure (mm Hg)                   |
| chol      | Continuous  | Serum cholesterol (mg/dl)                        |
| fbs       | Binary      | Fasting blood sugar > 120 mg/dl                  |
| restecg   | Categorical | Resting ECG results (0, 1, 2)                    |
| thalach   | Continuous  | Max heart rate achieved                          |
| exang     | Binary      | Exercise induced angina                          |
| oldpeak   | Continuous  | ST depression induced by exercise                |
| slope     | Categorical | Slope of peak exercise ST segment                |
| ca        | Ordinal     | Number of major vessels (0–3)                    |
| thal      | Categorical | Thalassemia (1=normal, 2=fixed defect, 3=reversible) |
| target    | Binary      | **0 = No Disease, 1 = Disease**                  |
