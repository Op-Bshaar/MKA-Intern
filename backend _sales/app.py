# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins, or use ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained model
model = joblib.load("sales_model.pkl")

# Define the feature order exactly as in training
FEATURES = [
    'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday',
    'Year', 'Month', 'Day',
    'CompetitionDistance', 'Promo2',
    'lag_1','lag_7','lag_30'
]

# Define request schema
class SalesRequest(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    SchoolHoliday: int
    Year: int
    Month: int
    Day: int
    CompetitionDistance: float
    Promo2: int
    lag_1: float
    lag_7: float
    lag_30: float

@app.post("/predict")
def predict_sales(data: SalesRequest):
    # Build DataFrame with correct column order
    X = pd.DataFrame([[
        data.Store,
        data.DayOfWeek,
        data.Promo,
        data.SchoolHoliday,
        data.Year,
        data.Month,
        data.Day,
        data.CompetitionDistance,
        data.Promo2,
        data.lag_1,
        data.lag_7,
        data.lag_30
    ]], columns=FEATURES)

    # Predict
    prediction = model.predict(X)[0]

    return {"predicted_sales": float(prediction)}
