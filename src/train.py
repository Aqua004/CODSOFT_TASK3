import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/churn.csv')
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    if 'Churn' not in data:
        parser.error('Missing Churn label; adapt the script for your dataset')
    labels = data.pop('Churn').astype(str).str.strip().str.lower()
    if not set(labels.unique()).issubset({'yes', 'no'}) or labels.nunique() != 2:
        parser.error('Churn must contain both Yes and No')
    y = labels.map({'no': 0, 'yes': 1})
    data = data.drop(columns=['customerID'], errors='ignore')
    if 'TotalCharges' in data:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    numeric = data.select_dtypes(include='number').columns.tolist()
    categorical = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    if not numeric and not categorical:
        parser.error('No usable features found')
    x_train, x_test, y_train, y_test = train_test_split(
        data[numeric + categorical], y, stratify=y, test_size=0.2, random_state=42
    )
    preprocess = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ])
    model = Pipeline([('preprocess', preprocess), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))])
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]
    metrics = {
        'roc_auc': roc_auc_score(y_test, scores),
        'average_precision': average_precision_score(y_test, scores),
        'confusion_matrix': confusion_matrix(y_test, predicted).tolist(),
        'classification_report': classification_report(y_test, predicted, output_dict=True, zero_division=0),
    }
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    joblib.dump(model, 'models/churn_model.joblib')
    Path('results/metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(classification_report(y_test, predicted, zero_division=0))
    print(f'ROC-AUC: {metrics["roc_auc"]:.4f}')


if __name__ == '__main__':
    main()
