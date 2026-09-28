# Bank Customer Churn Prediction (CodSoft Task 3)

Predict whether a bank customer leaves using the bank churn dataset linked in the CodSoft PDF: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction . This project does not claim any measured score until you run it.

## Dataset
Download the linked dataset and put its CSV into `data/`. The script accepts `Exited` (0/1), `churn` (0/1), or `Churn` (Yes/No or 0/1) as the target; pass the actual file name with `--data`. Bank-style columns such as `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, and `EstimatedSalary` are used if present. Customer identifiers and names are excluded. Check your CSV headers before running. Do not commit customer data.

## Run
```bash
python -m venv .venv
# Activate the virtual environment
pip install -r requirements.txt
python src/train.py --data data/Churn_Modelling.csv
```
Replace the filename in the last command with your actual downloaded CSV filename. The script saves a model in `models/` and measured metrics in `results/` locally; both are gitignored. It reports churn precision, recall, F1, ROC-AUC, average precision, and confusion matrix. Preprocessing is fit after the stratified holdout split.

## Submission
Run on your downloaded dataset, inspect the results and demonstrate the run on video. Add measured results here only after verifying them. If the dataset has a different label or structure, adapt the parser before claiming results.

## Results
Not yet run or independently verified.
