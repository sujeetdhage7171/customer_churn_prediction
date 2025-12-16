from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/model.pkl")

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
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Apply SAME encoding as training
    df["gender_Male"] = df["gender"].apply(lambda x: 1 if x == "Male" else 0)

    # Drop original categorical column
    df = df.drop(columns=["gender"])

    prediction = model.predict(df)[0]
    return {"churn": int(prediction)}
