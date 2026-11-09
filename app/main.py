"""
FastAPI service for loan-default probability prediction.
Loads a sklearn Pipeline (DictVectorizer + classifier) from pickle.
Uses Pydantic schemas for request/response validation.
"""

import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.schemas import LoanApplication, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "model.pkl"

app = FastAPI(
    title="Loan Default Prediction API",
    description="Predict the probability that a loan application will default.",
    version="1.0.0",
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")


def application_to_dict(app_data: LoanApplication) -> dict:
    emp = (
        app_data.employment_duration
        if app_data.employment_duration is not None
        else 1.0
    )
    rate = app_data.loan_int_rate if app_data.loan_int_rate is not None else 0.0
    return {
        "customer_age": app_data.customer_age,
        "customer_income": app_data.customer_income,
        "home_ownership": app_data.home_ownership,
        "employment_duration": emp,
        "loan_intent": app_data.loan_intent,
        "loan_grade": app_data.loan_grade,
        "loan_amnt": app_data.loan_amnt,
        "loan_int_rate": rate,
        "term_years": app_data.term_years,
        "historical_default": app_data.historical_default,
        "cred_hist_length": app_data.cred_hist_length,
        "loan_to_income": app_data.loan_amnt / (app_data.customer_income + 1),
        "income_per_year_emp": app_data.customer_income / (emp + 1),
    }


@app.get("/")
def root():
    return {
        "message": "Loan Default Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = application_to_dict(application)
        prob = float(model.predict_proba([features])[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    if prob >= 0.5:
        prediction = "DEFAULT"
        risk = "HIGH" if prob >= 0.7 else "MEDIUM"
    else:
        prediction = "NO DEFAULT"
        risk = "LOW" if prob < 0.3 else "MEDIUM"

    # Response is validated by PredictionResponse
    return PredictionResponse(
        default_probability=round(prob, 4),
        prediction=prediction,
        risk_level=risk,
    )
