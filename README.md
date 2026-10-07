# Bank Customer Churn Prediction (CodSoft Task 3)

Predict whether a bank customer leaves using the dataset linked in the CodSoft task brief: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

## Dataset
Download the dataset and place its CSV in `data/`. This project supports a binary target column named `Exited`, `churn`, or `Churn`. The completed run used `data/Churn_Modelling.csv`. Customer IDs, row identifiers, and surnames are excluded from features. Do not commit customer-level records.

## Methodology
- Preprocessing: median imputation and scaling for numeric variables; most-frequent imputation and one-hot encoding for categorical variables.
- Model: class-weighted Logistic Regression.
- Evaluation: stratified 80/20 train/test split with `random_state=42`.
- Metrics: churn-class precision, recall, F1, ROC-AUC, average precision, and confusion matrix.

## Run
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py --data data/Churn_Modelling.csv
```

## Verified Results
The project was run locally on the downloaded dataset.

- Test samples: 2,000
- Accuracy: 0.71
- Churn precision: 0.39
- Churn recall: 0.70
- Churn F1-score: 0.50
- Macro F1-score: 0.65
- Weighted F1-score: 0.74

The model identified 70% of actual churned customers in the holdout test set. Its churn precision of 0.39 indicates that some customers predicted to churn did not actually churn. In a retention setting, this baseline favours finding potential churners, but threshold selection and richer feature engineering could reduce unnecessary interventions.

## Output
Training saves `models/churn_model.joblib` and `results/metrics.json` locally. Both are excluded from Git.
