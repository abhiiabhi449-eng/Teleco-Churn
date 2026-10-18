# Telco Customer Churn Prediction

Predicts which telecom customers are likely to leave, using the Telco Customer Churn dataset (7,043 customers, 21 columns).

## Findings
- Month-to-month customers churn at 42.7%, versus 2.8% for two-year contracts.
- Churned customers have much shorter tenure.

## Method
- Cleaned TotalCharges (11 blank values for brand-new customers set to 0).
- One-hot encoded categorical columns, 80/20 stratified train/test split.
- Compared Logistic Regression (scaled) and Random Forest.

## Results
| Model | Accuracy | Churn recall |
|---|---|---|
| Logistic Regression | 0.807 | 0.57 |
| Random Forest | 0.793 | 0.51 |

Lowering the decision threshold to 0.4 raises churn recall to 0.67 at the cost of more false alarms.

## How to run
    pip install -r requirements.txt
    python src/train.py

## Next steps
Django REST API, Docker, deployment.
