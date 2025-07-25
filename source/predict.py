# source/predict.py
from fastapi import FastAPI, Request
import joblib
import pandas as pd

app = FastAPI()

# Load model at startup
model = joblib.load("models/model.pkl")

@app.post("/predict")
async def predict(request: Request):
    input_data = await request.json()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return {"churn": int(prediction)}