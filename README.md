# Loan Default Prediction – ML Zoomcamp Midterm Project

Binary classification project that predicts whether a loan application will **default**.

## Problem

Lenders need an automated, real-time risk score for new applications.  
A model that outputs a **default probability** can power:

- Instant approve / review / reject decisions
- Portfolio monitoring and stress testing
- Risk-based pricing

## Dataset

`data/loan_dataset.csv` (~32k historical applications) with features:

| Feature | Description |
|---------|-------------|
| customer_age | Age of applicant |
| customer_income | Annual income |
| home_ownership | RENT / OWN / MORTGAGE / OTHER |
| employment_duration | Years employed |
| loan_intent | Purpose of the loan |
| loan_grade | Credit grade A–E |
| loan_amnt | Requested amount |
| loan_int_rate | Interest rate (%) |
| term_years | Loan term |
| historical_default | Prior default flag |
| cred_hist_length | Length of credit history |

Target: `Current_loan_status` → `DEFAULT` / `NO DEFAULT` (~21 % positive class).

## Project structure

```
loan-default-project/
├── data/                    # raw CSV
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   └── train.py             # training script (notebook → script)
├── model/                   # persisted model + metadata (created by train.py)
├── app/
│   └── main.py              # FastAPI service
├── pyproject.toml           # project + dependencies
├── uv.lock                  # locked dependency versions (uv)
├── requirements.txt         # exported from uv.lock (for Docker)
├── Dockerfile
├── fly.toml
└── README.md
```

## 1. Setup (with uv + lockfile)

Install [uv](https://docs.astral.sh/uv/) if you don't have it yet:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment and sync **exact** versions from `uv.lock`:

```bash
uv sync --default-index https://pypi.org/simple
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

`uv sync` reads `pyproject.toml` + `uv.lock` and installs the locked dependency set into `.venv`.

If dependency downloads fail because an old package mirror is unreachable, refresh the lockfile against the official PyPI index and sync again:

```bash
uv lock --upgrade --default-index https://pypi.org/simple
uv sync --default-index https://pypi.org/simple
```


If you only need production packages (no project install):

```bash
uv venv
uv pip sync requirements.txt
source .venv/bin/activate
```

## 2. Train the model

```bash
uv run python src/train.py
# or after activating: python src/train.py
```

This will:

- Clean the data (currency symbols, age outliers, missing values)
- Engineer `loan_to_income` and `income_per_year_emp`
- Train **four** classification models inside `Pipeline(DictVectorizer → model)` and compare them:
  1. Logistic Regression
  2. Random Forest
  3. Gradient Boosting
  4. HistGradient Boosting
- Tune the best model family with `RandomizedSearchCV`
- Save the winning pipeline with **pickle** to `model/model.pkl`

Expected test ROC-AUC ≈ 0.90+.

## 3. Run the API locally

Request and response bodies are validated with **Pydantic** (`app/schemas.py`). The test client (`scripts/test_api.py`) validates both sides before/after the HTTP call.


```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "cred_hist_length": 4
  }'
```

## 4. Docker (local)

```bash
# Build (model must already exist)
docker build -t loan-default-api .

# Run
docker run -p 8000:8000 loan-default-api
```

## 5. Deploy to Fly.io

1. Install the Fly CLI and log in: https://fly.io/docs/hands-on/install-flyctl/
2. From the project root:

```bash
fly launch --name loan-default-api --region iad --no-deploy   # first time only
# or just use the provided fly.toml
fly deploy
```

3. Open the app:

```bash
fly open
# or
fly status
```

The service will be available at `https://loan-default-api.fly.dev` (or the name you chose).

## Model performance (indicative)

| Model                  | CV AUC   | Test AUC |
|------------------------|----------|----------|
| Logistic Regression    | ~0.88    | ~0.88    |
| Random Forest          | ~0.91    | ~0.91    |
| Gradient Boosting      | ~0.92    | ~0.92    |
| HistGradient Boosting  | ~0.92    | ~0.92    |

The best model is tuned further with `RandomizedSearchCV` and saved as the final artifact.

## Evaluation criteria covered

- Interesting problem + clear business use-case
- Thorough EDA & feature analysis
- Four classification models + tuning + model selection
- Notebook exported to production script (`src/train.py`)
- FastAPI web service
- Local Docker deployment
- Cloud deployment instructions (Fly.io)

## License

Educational project for ML Zoomcamp.
