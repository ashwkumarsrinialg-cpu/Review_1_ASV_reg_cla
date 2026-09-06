# 23CSE301 Machine Learning — Capstone Project

## Overview

This repository contains the **Review 1** implementation for the 23CSE301 Machine Learning Capstone Project (Academic Year 2026–27).

The project currently covers:

- **Full Regression Track** using the California Housing dataset.
- **Classification Track — Part A** using the Adult / Census Income dataset.

The notebooks implement the end-to-end machine learning workflow required by the capstone: dataset auditing, exploratory data analysis (EDA), data cleaning, preprocessing, feature engineering, model training, model comparison, evaluation, hyperparameter tuning, diagnostics, and model serialization.

> **Review 1 scope:** The official capstone guidelines require the complete Regression Track and the first five Classification algorithms. Classification Part B and the Clustering Track are planned for Review 2.

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── housing.csv
│   └── adult.csv
├── notebooks/
│   ├── regression.ipynb
│   └── classification.ipynb
└── models/
    ├── regression_best_model.pkl
    └── classification_best_model.pkl
```

The `models/` directory is created automatically by the notebooks when the best models are saved.

---

# 1. Regression Track

## Problem Statement

The regression task predicts the **median house value (`median_house_value`)** for California housing block groups using demographic, geographic, and housing-related attributes.

### Dataset

**California Housing Dataset**

- Records: **20,640**
- Raw columns: **10**
- Features: **9 numerical + 1 categorical**
- Target: `median_house_value`
- Target range in the dataset: approximately **$14,999 to $500,001**
- Categorical feature: `ocean_proximity`
- Missing values: `total_bedrooms` contains **207 missing values**

The missing numerical values are handled using median imputation within the preprocessing pipeline.

## Exploratory Data Analysis

The regression notebook performs:

- Dataset shape and data-type inspection
- Missing-value audit
- Summary statistics
- Target distribution analysis
- Feature distribution/skewness analysis
- Feature-target relationship visualizations
- Ocean-proximity versus house-value analysis
- Correlation analysis

The EDA is accompanied by written observations describing the patterns revealed by the visualizations.

## Preprocessing

The regression pipeline uses:

- **80:20 train/test split**
- Median imputation for numerical missing values
- `StandardScaler` for numerical features
- `OneHotEncoder(drop='first')` for `ocean_proximity`
- Preprocessing fitted on the training data to avoid data leakage

## Feature Engineering

Three ratio-based features are created:

```text
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households
```

These features provide additional information about household size, room composition, and population density.

## Algorithms

All **10 required regression algorithms** are implemented:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor (SVR)
10. K-Nearest Neighbors Regressor

All models are evaluated using the **same held-out test set**.

## Regression Results

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.6353 | $69,127.04 | $49,645.49 |
| Ridge Regression | 0.6352 | $69,136.26 | $49,653.25 |
| Lasso Regression | 0.6348 | $69,179.21 | $49,648.56 |
| ElasticNet Regression | 0.6287 | $69,757.85 | $50,091.85 |
| Polynomial Regression | 0.6209 | $70,480.08 | $44,709.93 |
| Decision Tree Regressor | 0.7073 | $61,928.77 | $41,936.01 |
| Random Forest Regressor | 0.7963 | $51,668.88 | $33,800.13 |
| Gradient Boosting Regressor | 0.7807 | $53,612.12 | $36,523.02 |
| SVR | -0.0871 | $119,355.55 | $101,782.44 |
| KNN Regressor | 0.7120 | $61,429.77 | $40,669.99 |

### Initial Best Model

The initial benchmark identified **Random Forest Regressor** as the best model based on test-set R²:

- **R²:** 0.7963
- **RMSE:** $51,668.88
- **MAE:** $33,800.13

## Cross-Validation and Hyperparameter Tuning

The two strongest ensemble models, Random Forest and Gradient Boosting, were evaluated using 5-fold cross-validation.

| Model | Mean CV R² | CV Std. |
|---|---:|---:|
| Random Forest | 0.8006 | 0.0047 |
| Gradient Boosting | 0.7912 | 0.0057 |

Both models were then tuned using `GridSearchCV`.

### Tuned Random Forest

Best parameters:

```text
max_depth = None
min_samples_split = 2
n_estimators = 150
```

Performance:

- Best 5-fold CV R²: **0.8113**
- Test R²: **0.8121**
- Test RMSE: **$49,616.99**
- Test MAE: **$31,861.38**

### Tuned Gradient Boosting

Best parameters:

```text
learning_rate = 0.1
max_depth = 6
n_estimators = 150
```

Performance:

- Best 5-fold CV R²: **0.8326**
- Test R²: **0.8366**
- Test RMSE: **$46,267.30**
- Test MAE: **$30,189.77**

### Final Regression Model

The final regression model is the **Tuned Gradient Boosting Regressor**, selected using the highest test-set R².

**Final performance:**

- **R²:** 0.8366
- **RMSE:** $46,267.30
- **MAE:** $30,189.77

The notebook also generates:

- Residual plot
- Predicted-vs-actual plot
- Feature-importance visualization
- Diagnostic statistics

The most influential feature in the fitted model is `median_income`, followed by `ocean_proximity_INLAND` and `population_per_household`.

---

# 2. Classification Track — Part A

## Problem Statement

The classification task predicts whether an individual's annual income is:

- `<=50K`
- `>50K`

using the **Adult / Census Income dataset**.

### Dataset

- Records: **32,561**
- Raw columns: **15**
- Target: `income`
- Binary classification problem
- Approximately **76%** of observations belong to `<=50K` and **24%** to `>50K`

Important features include:

- `age`
- `workclass`
- `education`
- `education.num`
- `marital.status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `capital.gain`
- `capital.loss`
- `hours.per.week`
- `native.country`

Missing values are represented by `"?"` in `workclass`, `occupation`, and `native.country`.

## Exploratory Data Analysis

The classification notebook performs:

- Dataset shape and data-type inspection
- Missing-value audit
- Target class-balance analysis
- Feature distributions
- Income versus education analysis
- Hours-per-week versus income analysis
- Numerical correlation analysis

Key observations include:

- Higher educational attainment is associated with a greater proportion of individuals earning `>50K`.
- Individuals earning `>50K` tend to have higher median working hours per week.

## Preprocessing

The classification pipeline uses:

- Missing-value conversion from `"?"` to `NaN`
- Mode imputation for categorical missing values
- Median imputation for numerical values
- IQR-based capping for extreme numerical observations
- One-hot encoding for categorical variables
- `StandardScaler` for numerical features
- **80:20 stratified train/test split**
- `random_state=42`

All preprocessing transformations are fitted using the training data.

## Feature Engineering

Two additional features are created:

```text
Capital_Net = capital.gain - capital.loss
Has_Capital_Gain = binary indicator
```

These features capture the net effect of capital gains/losses and whether an individual has any capital gain.

## Algorithms

The five required **Classification Part A** algorithms are:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Classifier (SVC)

## Classification Results

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Support Vector Classifier (SVC) | 0.8360 | 0.8273 | 0.8360 | **0.8284** | 0.8673 |
| Logistic Regression | 0.8348 | 0.8265 | 0.8348 | 0.8281 | **0.8832** |
| KNN (k=7) | 0.8115 | 0.8097 | 0.8115 | 0.8106 | 0.8486 |
| Decision Tree (max_depth=4) | 0.8213 | 0.8111 | 0.8213 | 0.8003 | 0.8620 |
| Gaussian Naive Bayes | 0.5045 | 0.8017 | 0.5045 | 0.5159 | 0.7010 |

### Best Classification Model

The notebook selects the model with the highest **Weighted F1-score**.

The best Part A model is:

**Support Vector Classifier (SVC)**

- Accuracy: **0.8360**
- Weighted Precision: **0.8273**
- Weighted Recall: **0.8360**
- Weighted F1: **0.8284**
- ROC-AUC: **0.8673**

For the SVC, the notebook uses the model's decision function for ROC analysis rather than probability estimates.

### Best Model Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| `<=50K` | 0.8656 | 0.9281 | 0.8958 |
| `>50K` | 0.7069 | 0.5459 | 0.6160 |

The classification notebook also provides:

- Confusion matrices for all five models
- Overlayed ROC curves
- Decision tree visualization
- Model performance comparison chart
- Detailed classification report for the selected model

---

# 3. Model Selection Summary

| Track | Selected Model | Main Selection Criterion | Test Performance |
|---|---|---|---|
| Regression | Tuned Gradient Boosting Regressor | Highest test R² after tuning | R² = **0.8366** |
| Classification Part A | Support Vector Classifier (SVC) | Highest Weighted F1 | F1 = **0.8284** |

---

# 4. Installation and Execution

## Requirements

- Python **3.9+**
- Jupyter Notebook / JupyterLab
- Dependencies listed in `requirements.txt`

## 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the notebooks

```bash
jupyter notebook notebooks/regression.ipynb
```

and

```bash
jupyter notebook notebooks/classification.ipynb
```

Run each notebook **from top to bottom** so that all preprocessing, model training, evaluation, visualizations, and model-saving cells execute correctly.

The notebooks save the selected models under:

```text
models/
```

---

# 5. Reproducibility and Data Leakage Prevention

The project uses `random_state=42` wherever applicable to make experiments reproducible.

Preprocessing is performed carefully to avoid data leakage:

- Train/test splitting is performed before fitting preprocessing transformations.
- Numerical imputers and scalers are fitted using training data.
- Categorical encoders are fitted using training data.
- The same preprocessing transformation is then applied to the test data.

The same held-out test split is used across the models within each track so that model comparisons remain fair.

---

# 6. Capstone Requirements Coverage

### Review 1

| Requirement | Status |
|---|---|
| Dataset audit and EDA | Completed |
| Missing-value handling | Completed |
| Encoding and scaling | Completed |
| Feature engineering | Completed |
| All 10 regression algorithms | Completed |
| Regression comparison table | Completed |
| Regression 5-fold CV | Completed |
| Regression hyperparameter tuning | Completed |
| Regression diagnostics | Completed |
| Classification Part A — 5 algorithms | Completed |
| Classification comparison table | Completed |
| Classification confusion matrices | Completed |
| Classification ROC curves / ROC-AUC | Completed |
| Classification decision-tree visualization | Completed |

The Review 1 rubric allocates marks to Dataset & EDA, Preprocessing & Feature Engineering, the Regression Track, Classification Part A, Presentation, and Viva.

### Review 2 — Planned Work

According to the capstone guidelines, the remaining work includes:

- Classification Part B:
  - Random Forest Classifier
  - AdaBoost Classifier
  - Gradient Boosting Classifier
  - Bagging Classifier
  - MLP Classifier
- Consolidated comparison of all **10 classification algorithms**
- Classification hyperparameter tuning and final model selection
- Full Clustering Track:
  - K-Means
  - Agglomerative Hierarchical Clustering
- Clustering evaluation using:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
- Required clustering visualizations:
  - Elbow curve
  - Dendrogram
  - PCA 2D visualization
- t-SNE visualization where applicable
- Final integrated README and repository documentation

---

# 7. Future Extensions

The capstone guidelines also allow optional bonus work:

- **Interactive GUI** using Streamlit or Gradio
- **Public deployment** of the application

These can be added after the required Review 2 implementation.

---

# 8. Academic Integrity

All analysis, interpretation, and feature-engineering decisions in this repository are intended to be original to the team.

## References

- California Housing dataset used for the regression task.
- Adult / Census Income dataset used for the classification task.
- 23CSE301 Machine Learning Capstone Project Guidelines, Algorithm List & Evaluation Rubrics, Academic Year 2026–27.

---

## Authors

**23CSE301 Machine Learning Capstone Project**

Ashwin Kumar Srinivasan -CB.SC.U4CSE24607-Amrita University -Coimbatore
Salla Sai Sumallik -CB.SC.U4CSE24647-Amrita University -Coimbatore
Tallada Vamshidhar -CB.SC.U4CSE24660-Amrita University -Coimbatore
B.Tech. Computer Science and Engineering — III Year  
Academic Year 2026–27
