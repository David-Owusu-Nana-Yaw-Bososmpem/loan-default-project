#!/usr/bin/env python3
"""
Smoke-test the prediction API with Pydantic validation.

Validates:
  - request payload (LoanApplication)
  - response body (PredictionResponse)
"""

import json
import sys
from pathlib import Path

import requests
from pydantic import ValidationError

# Allow importing app.schemas when run from scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.schemas import LoanApplication, PredictionResponse  # noqa: E402

URL = "http://localhost:8000/predict"

# --- Valid request (Pydantic validates fields & enums) ---
payload = {
    "customer_age": 28,
    "customer_income": 55000,
    "home_ownership": "RENT",
    "employment_duration": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 15000,
    "loan_int_rate": 12.5,
    "term_years": 5,
    "historical_default": 0,
    "cred_hist_length": 4,
}

print("Validating request with Pydantic …")
try:
    request_model = LoanApplication(**payload)
except ValidationError as e:
    print("Request validation failed:")
    print(e)
    sys.exit(1)

print("Request OK:", request_model.model_dump())

print(f"\nPOST {URL}")
try:
    r = requests.post(URL, json=request_model.model_dump(), timeout=10)
except requests.exceptions.ConnectionError:
    print(
        "Connection refused — start the API first:\n"
        "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    )
    sys.exit(1)

print("Status:", r.status_code)
print("Raw body:", r.text)

if r.status_code != 200:
    print("API returned an error.")
    sys.exit(1)

print("\nValidating response with Pydantic …")
try:
    response_model = PredictionResponse(**r.json())
except ValidationError as e:
    print("Response validation failed:")
    print(e)
    sys.exit(1)

print("Response OK:")
print(json.dumps(response_model.model_dump(), indent=2))

# --- Invalid request example (should fail local validation) ---
print("\n--- Negative test: invalid loan_grade ---")
bad = payload.copy()
bad["loan_grade"] = "Z"
try:
    LoanApplication(**bad)
    print("Unexpected: invalid grade was accepted")
except ValidationError as e:
    print("Caught expected ValidationError:")
    print(e)
