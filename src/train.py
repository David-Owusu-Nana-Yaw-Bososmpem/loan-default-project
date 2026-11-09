#!/usr/bin/env python3
"""
Train loan-default models using sklearn Pipeline + DictVectorizer.
Compares 4 classifiers, tunes the best, saves with pickle.

Usage:
    python src/train.py
    uv run python src/train.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "loan_dataset.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Features used as dict keys for DictVectorizer
FEATURE_COLS = [
    "customer_age",
    "customer_income",
    "home_ownership",
    "employment_duration",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "term_years",
    "historical_default",
    "cred_hist_length",
    "loan_to_income",
    "income_per_year_emp",
]


def clean_loan_amount(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("£", "").replace(",", "").replace('"', "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_income(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "").replace('"', "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["loan_amnt"] = df["loan_amnt"].apply(clean_loan_amount)
    df["customer_income"] = df["customer_income"].apply(clean_income)
    df = df.dropna(subset=["Current_loan_status"])
    df["default"] = (df["Current_loan_status"] == "DEFAULT").astype(int)
    df["historical_default"] = (
        df["historical_default"].map({"Y": 1, "N": 0}).fillna(0).astype(int)
    )
    df = df[(df["customer_age"] >= 18) & (df["customer_age"] <= 100)]
    df.loc[df["employment_duration"] > 50, "employment_duration"] = np.nan
    # Fill remaining numeric nulls with median before dict conversion
    for col in ["employment_duration", "loan_int_rate", "loan_amnt", "customer_income"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df = df.drop(columns=["customer_id", "Current_loan_status"], errors="ignore")
    df["loan_to_income"] = df["loan_amnt"] / (df["customer_income"] + 1)
    df["income_per_year_emp"] = df["customer_income"] / (
        df["employment_duration"].fillna(1) + 1
    )
    return df


def to_dicts(df: pd.DataFrame) -> list[dict]:
    """Convert rows to list of feature dicts for DictVectorizer."""
    records = df[FEATURE_COLS].to_dict(orient="records")
    # Ensure categorical values are strings; numeric stay numeric
    for row in records:
        for key in ("home_ownership", "loan_intent", "loan_grade"):
            if key in row and row[key] is not None:
                row[key] = str(row[key])
    return records


def main():
    print("Loading & cleaning data …")
    df = load_and_clean(DATA_PATH)
    print(f"Shape: {df.shape}, default rate: {df['default'].mean():.2%}")

    y = df["default"].values
    dicts = to_dicts(df)

    # Stratified split on indices
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )
    train_dicts = [dicts[i] for i in train_idx]
    test_dicts = [dicts[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # ----- Four models, each as Pipeline(DictVectorizer -> classifier) -----
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, max_depth=6, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    print("\n=== Four-model comparison (Pipeline + DictVectorizer) ===")
    for name, clf in models.items():
        pipe = Pipeline(
            steps=[
                ("dv", DictVectorizer(sparse=False)),
                ("clf", clf),
            ]
        )
        pipe.fit(train_dicts, y_train)
        y_prob = pipe.predict_proba(test_dicts)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        results[name] = {"test_auc": float(auc), "test_ap": float(ap), "pipeline": pipe}
        print(f"{name:22s}  Test AUC={auc:.4f}  AP={ap:.4f}")

    best_name = max(results, key=lambda k: results[k]["test_auc"])
    print(f"\nBest baseline: {best_name}")

    # ----- Tune best model family -----
    if best_name in ("GradientBoosting", "HistGradientBoosting"):
        tune_clf = HistGradientBoostingClassifier(random_state=42)
        param_dist = {
            "clf__max_iter": [100, 150, 200],
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.05, 0.1, 0.15],
            "clf__min_samples_leaf": [10, 20, 30],
        }
        tune_label = "HistGradientBoosting"
    elif best_name == "RandomForest":
        tune_clf = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        param_dist = {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [8, 12, 16],
            "clf__min_samples_leaf": [2, 5, 10],
        }
        tune_label = "RandomForest"
    else:
        tune_clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )
        param_dist = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
        tune_label = "LogisticRegression"

    print(f"Tuning {tune_label} …")
    tune_pipe = Pipeline(
        steps=[
            ("dv", DictVectorizer(sparse=False)),
            ("clf", tune_clf),
        ]
    )
    search = RandomizedSearchCV(
        tune_pipe,
        param_distributions=param_dist,
        n_iter=12,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(train_dicts, y_train)

    final_model = search.best_estimator_
    y_prob = final_model.predict_proba(test_dicts)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    test_ap = average_precision_score(y_test, y_prob)

    print("Best params:", search.best_params_)
    print(f"CV best AUC : {search.best_score_:.4f}")
    print(f"Test AUC    : {test_auc:.4f}")
    print(f"Test AP     : {test_ap:.4f}")

    # ----- Save with pickle -----
    model_path = MODEL_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nModel (pipeline) saved → {model_path}")

    meta = {
        "models_compared": list(results.keys()),
        "baseline_results": {
            k: {"test_auc": v["test_auc"], "test_ap": v["test_ap"]}
            for k, v in results.items()
        },
        "best_baseline": best_name,
        "tuned_model": tune_label,
        "feature_cols": FEATURE_COLS,
        "best_params": {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in search.best_params_.items()
        },
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "model_file": "model.pkl",
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        import json

        json.dump(meta, f, indent=2)
    print("Metadata saved.")


if __name__ == "__main__":
    main()
