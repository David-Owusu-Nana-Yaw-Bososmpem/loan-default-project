"""Pydantic schemas for request/response validation."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LoanApplication(BaseModel):
    """Validated loan application features for prediction."""

    customer_age: int = Field(..., ge=18, le=100, examples=[28])
    customer_income: float = Field(..., gt=0, examples=[55000])
    home_ownership: str = Field(
        ...,
        examples=["RENT"],
        description="RENT | OWN | MORTGAGE | OTHER",
    )
    employment_duration: Optional[float] = Field(
        None, ge=0, examples=[5], description="Years of employment"
    )
    loan_intent: str = Field(
        ...,
        examples=["PERSONAL"],
        description="PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION",
    )
    loan_grade: str = Field(..., examples=["B"], description="A | B | C | D | E")
    loan_amnt: float = Field(..., gt=0, examples=[15000])
    loan_int_rate: Optional[float] = Field(
        None, ge=0, le=50, examples=[12.5], description="Interest rate in percent"
    )
    term_years: int = Field(..., ge=1, le=15, examples=[5])
    historical_default: int = Field(
        0, ge=0, le=1, examples=[0], description="1 if previous default, else 0"
    )
    cred_hist_length: int = Field(..., ge=0, examples=[4])

    @field_validator("home_ownership")
    @classmethod
    def validate_home_ownership(cls, v: str) -> str:
        allowed = {"RENT", "OWN", "MORTGAGE", "OTHER"}
        v_up = v.upper().strip()
        if v_up not in allowed:
            raise ValueError(f"home_ownership must be one of {sorted(allowed)}")
        return v_up

    @field_validator("loan_intent")
    @classmethod
    def validate_loan_intent(cls, v: str) -> str:
        allowed = {
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        }
        v_up = v.upper().strip()
        if v_up not in allowed:
            raise ValueError(f"loan_intent must be one of {sorted(allowed)}")
        return v_up

    @field_validator("loan_grade")
    @classmethod
    def validate_loan_grade(cls, v: str) -> str:
        allowed = {"A", "B", "C", "D", "E"}
        v_up = v.upper().strip()
        if v_up not in allowed:
            raise ValueError(f"loan_grade must be one of {sorted(allowed)}")
        return v_up


class PredictionResponse(BaseModel):
    """Validated prediction response."""

    default_probability: float = Field(..., ge=0.0, le=1.0)
    prediction: str
    risk_level: str

    @field_validator("prediction")
    @classmethod
    def validate_prediction(cls, v: str) -> str:
        allowed = {"DEFAULT", "NO DEFAULT"}
        if v not in allowed:
            raise ValueError(f"prediction must be one of {allowed}")
        return v

    @field_validator("risk_level")
    @classmethod
    def validate_risk(cls, v: str) -> str:
        allowed = {"LOW", "MEDIUM", "HIGH"}
        if v not in allowed:
            raise ValueError(f"risk_level must be one of {allowed}")
        return v
