from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd

class CustomerData(BaseModel):
    customer_id: int
    credit_score: float
    country: str
    gender: str
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float

    class Config:
        schema_extra = {
            "example": {
                "customer_id": 19764529,
                "credit_score": 690,
                "country": "France",
                "gender": "Female",
                "age": 35,
                "tenure": 5,
                "balance": 20.00,
                "products_number": 2,
                "credit_card": 1,
                "active_member": 0,
                "estimated_salary": 90000.05
            }
        }

app = FastAPI()

model = joblib.load("Bank_customer_churn_pipeline_model.joblib")


@app.post("/predict")
def predict(data: CustomerData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict using pipeline
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": round(probability, 4)
    }

