import json
from pathlib import Path

import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "churn_model.pkl")
with open(BASE_DIR / "models" / "columns.json") as f:
    columns = json.load(f)


def hello(request):
    return JsonResponse({"message": "Hello from the churn API"})


@csrf_exempt
def predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "Send a POST request"}, status=405)

    data = json.loads(request.body)

    row = pd.DataFrame([data]).reindex(columns=columns, fill_value=0)

    probability = model.predict_proba(row)[0][1]
    return JsonResponse({
        "churn_probability": round(float(probability), 3),
        "prediction": int(probability >= 0.4),
    })