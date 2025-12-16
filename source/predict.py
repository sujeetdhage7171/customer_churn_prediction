from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("models/model.pkl")

# EXACT feature order used during training
FEATURE_COLUMNS = [
    "age",
    "income",
    "subscription_length",
    "gender_Male",
]

class ChurnInput(BaseModel):
    age: int
    income: int
    gender: str
    subscription_length: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: ChurnInput):
    # Build row exactly as training expected
    row = {
        "age": data.age,
        "income": data.income,
        "subscription_length": data.subscription_length,
        "gender_Male": 1 if data.gender == "Male" else 0,
    }

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    prediction = model.predict(df)[0]
    return {"churn": int(prediction)}
