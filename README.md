# Customer Churn Prediction (CodSoft Task 3)

Predict whether a customer will churn with a preprocessing pipeline and logistic regression. This implementation expects the IBM-style Telco Customer Churn dataset with `Churn` as Yes/No and optionally `TotalCharges`, `customerID`, and `tenure`. Adjust the parser if your CodSoft download differs.

## Dataset
Download the dataset through the Task 3 link in the internship PDF. Save it as `data/churn.csv`; do not commit the raw file.

## Run
```bash
python -m venv .venv
# Activate the environment
pip install -r requirements.txt
python src/train.py --data data/churn.csv
```

The script writes `results/metrics.json` and `models/churn_model.joblib`, both excluded from git. It reports churn precision/recall/F1, ROC-AUC, average precision, and a confusion matrix. Split before fitting imputers, encoders, and scaler. Results depend on the data version.

## Demo / submission
Show a run, explain which customer attributes were used, report actual holdout metrics, and discuss false negatives. Do not claim the model is ready for real customer decisions without further validation.

## Results
Not yet run or independently verified.
