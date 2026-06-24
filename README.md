# IEEE-CIS Fraud Detection

Kaggle competition — Binary classification to predict whether a transaction is fraudulent, based on 590,000 transactions and 400+ features (transaction amount, card info, device, browser, behavioral signals).

🔗 [Competition link](https://www.kaggle.com/competitions/ieee-fraud-detection)

---

## Results

| Metric | Score |
|---|---|
| Cross-validation AUC (5 folds) | 0.9596 |
| Kaggle public leaderboard | 0.9293 |
| Kaggle private leaderboard | 0.8921 |

---

## Project Structure

```
ieee-fraud-detection/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_EDA_transaction.ipynb  # Exploratory analysis — transaction data
│   ├── 02_EDA_identity.ipynb      # Exploratory analysis — identity data
│   ├── 03_model.ipynb             # XGBoost pipeline, Optuna tuning, results
│   └── 03_temporal_model.ipynb    # TimeSeriesPlit pipeline,results
├── src/
│   ├── preclean.py                # Full preprocessing pipeline
│   └── func.ipynb                 # Function for the EDA
└── outputs/
    ├── feature_Importance.py     # Most important features with XGBoost
    └── submission.csv            # Results with transactions and isFraud
```

---

## Approach

### 1. Exploratory Data Analysis
Conducted a thorough EDA across both datasets (transaction + identity), with a focus on understanding the data structure before any modeling.

Key findings:
- **Missing values carry fraud signal** — NaN rates differ significantly between fraudulent and legitimate transactions across M, D and identity columns. Missing data was deliberately kept as a signal rather than blindly imputed.
- **Fraudsters hide their technical footprint** — transactions flagged as fraudulent tend to have more missing technical fields (device, browser, OS). A `count_missing_tech_block` feature captures this behavioral pattern.
- **Class imbalance is severe** — only 3.5% of transactions are fraudulent, requiring careful handling during modeling.
- **High redundancy in C variables** — PCA on the C group shows 2 components explain 91% of variance, suggesting strong correlation between counting variables.

### 2. Preprocessing (`src/preclean.py`)
- **Binary M columns** (M1–M9): mapped T → 1, F → 0
- **M4**: one-hot encoded (3 categories)
- **Email domains** (P_emaildomain, R_emaildomain): simplified to 5 groups (Google, Yahoo, Microsoft, Apple, Other) to handle high cardinality (~900 unique values)
- **id_33** (screen resolution): decomposed into width, height, and pixel count
- **OS / Browser**: simplified into families (Android, iOS, Windows, Chrome, Firefox, etc.)
- **DeviceInfo**: grouped by brand (Samsung, Apple, Motorola, etc.)
- Remaining categorical columns handled via one-hot encoding

XGBoost handles NaN values natively — float columns (V*, C*, D*) were intentionally left with NaN rather than imputed, preserving the fraud signal contained in missingness.

### 3. Modeling
- **Algorithm**: XGBoost Classifier
- **Validation**: Stratified K-Fold (5 folds) to preserve the 3.5% fraud ratio in each fold
- **Class imbalance**: handled via `scale_pos_weight = len(y[y==0]) / len(y[y==1]) ≈ 27`
- **Predictions**: averaged across the 5 fold models (ensembling) rather than retraining a separate final model

### 4. Hyperparameter Tuning (Optuna)
Bayesian optimization using Optuna's TPE sampler over 20 trials:

| Parameter | Search range |
|---|---|
| `max_depth` | [3, 9] |
| `learning_rate` | [0.01, 0.3] (log scale) |
| `subsample` | [0.5, 1.0] |
| `colsample_bytree` | [0.5, 1.0] |
| `min_child_weight` | [1, 20] |
| `reg_alpha` | [1e-3, 10.0] (log scale) |
| `reg_lambda` | [1e-3, 10.0] (log scale) |

The optimization converged after only **6 trials out of 15**, confirming the search space was well-targeted from the start.

### 5. Temporal Validation
Replaced StratifiedKFold with TimeSeriesSplit to better reflect
real-world conditions. Cross-validation AUC dropped from 0.96 to 0.88,
confirming that random splitting overestimated performance on this
chronological dataset. The temporal AUC (0.88) aligns closely with the
Kaggle private score (0.8921).

---

## Tech Stack

| Tool | Usage |
|---|---|
| Python 3.12 | Core language |
| Pandas / NumPy | Data manipulation |
| XGBoost | Gradient boosting model |
| Scikit-learn | Cross-validation, metrics |
| Optuna | Hyperparameter optimization |
| Matplotlib / Seaborn | Visualization |

---

## Future Improvements

- **Feature engineering** — aggregate features per card (average transaction amount, frequency of use) and anomaly ratio features (e.g. "this transaction is 5x the usual amount for this card")
---

## How to Run

```bash
# Clone the repo
git clone https://github.com/leadervieux/Fraud-project.git
cd ieee-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Download the data from Kaggle and place it in a data/ folder
# https://www.kaggle.com/competitions/ieee-fraud-detection/data

# Run notebooks in order
jupyter notebook notebooks/01_EDA_transaction.ipynb
```

---

## Notes

This project was built as part of a data science portfolio targeting quantitative and risk roles in banking and fintech.
The focus was on **methodological rigor** (EDA before modeling, honest evaluation, documented decisions) rather than leaderboard optimization.
