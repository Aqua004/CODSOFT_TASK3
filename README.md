## Results

The model was trained using a preprocessing pipeline and class-weighted Logistic Regression.

- Accuracy: 0.71
- Churn precision: 0.39
- Churn recall: 0.70
- Churn F1-score: 0.50
- Macro F1-score: 0.65
- Weighted F1-score: 0.74
- Test samples: 2,000

The model identifies 70% of churned customers. Its churn precision of 0.39 indicates that some customers predicted as likely to churn did not actually churn. Recall is important in this application because missing a potential churn customer may reduce the opportunity for retention action.