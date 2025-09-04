from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("Bank Customer Churn.joblib")

@app.post("/predict")
def predict(features: list):
    pred = model.predict([np.array(features)])
    return {"prediction": pred}
