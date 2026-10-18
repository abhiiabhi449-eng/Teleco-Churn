from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from prepare import load_and_prepare

df = load_and_prepare()

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("=====", name, "=====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    best = models["Logistic Regression"]
proba = best.predict_proba(X_test)[:, 1]

for threshold in [0.5, 0.4, 0.3]:
    y_pred = (proba >= threshold).astype(int)
    print("Threshold", threshold)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

import joblib
import os

os.makedirs("models", exist_ok=True)
joblib.dump(best, "models/churn_model.pkl")
print("Model saved")