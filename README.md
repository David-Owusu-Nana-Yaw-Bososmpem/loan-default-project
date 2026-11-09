# Loan Default Prediction API
### Machine Learning Zoomcamp Midterm Project

A machine learning classification project that predicts the probability of loan default using historical loan application data. The trained model is deployed as a REST API using FastAPI, Docker, and Fly.io.

**Live API:** https://loan-default-project.fly.dev

**Interactive API documentation:** https://loan-default-project.fly.dev/docs

**Health check:** https://loan-default-project.fly.dev/health

---

## 1. Project Overview

Lending institutions need reliable methods to assess the credit risk of loan applicants. This project develops a binary classification model that estimates the probability that a loan applicant will default.

The prediction can support:

- Credit risk assessment.
- Loan application screening.
- Risk-based pricing.
- Portfolio monitoring.
- Data-driven lending decisions.

The API accepts applicant and loan information and returns a default probability, a prediction, and a risk category.

## 2. Dataset

The project uses approximately 32,000 historical loan applications.

Dataset location:

`data/loan_dataset.csv`

The target variable is `Current_loan_status`, which distinguishes between DEFAULT and NO DEFAULT.

### Input features

| Feature | Description |
|---|---|
| customer_age | Applicant's age |
| customer_income | Annual income |
| home_ownership | Housing status |
| employment_duration | Years employed |
| loan_intent | Purpose of the loan |
| loan_grade | Credit grade (A–E) |
| loan_amnt | Requested loan amount |
| loan_int_rate | Interest rate |
| term_years | Loan repayment term |
| historical_default | Previous default indicator |
| cred_hist_length | Length of credit history |

Two additional features are engineered:

- `loan_to_income`
- `income_per_year_emp`

The target distribution has approximately 21% positive default cases.

## 3. Data Preparation and Exploratory Analysis

Data preparation includes:

- Cleaning numerical values and currency symbols.
- Handling missing values.
- Removing observations outside the selected applicant age range.
- Handling unrealistic employment durations.
- Converting historical default indicators into binary values.
- Engineering additional financial features.

The exploratory analysis and modelling process are documented in:

`notebooks/01_eda_and_modeling.ipynb`

## 4. Model Development

Four classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. HistGradientBoosting

Each classifier was implemented using a Scikit-learn Pipeline with DictVectorizer.

The dataset was divided into training and testing sets using an 80:20 stratified split with `random_state=42`.

### Baseline model comparison

| Model | Test ROC-AUC | Test Average Precision |
|---|---:|---:|
| Logistic Regression | 0.8319 | 0.6267 |
| Random Forest | 0.9273 | 0.8587 |
| Gradient Boosting | 0.9331 | 0.8685 |
| HistGradientBoosting | 0.9419 | 0.8860 |

### Hyperparameter tuning

HistGradientBoosting achieved the highest baseline ROC-AUC and was selected for further optimization.

RandomizedSearchCV was used with:

- 3-fold cross-validation.
- 12 randomized parameter combinations.
- ROC-AUC as the optimization metric.
- Random state of 42.

The selected parameters were:

| Parameter | Value |
|---|---:|
| max_iter | 150 |
| max_depth | 8 |
| learning_rate | 0.1 |
| min_samples_leaf | 20 |

### Final model performance

| Evaluation metric | Result |
|---|---:|
| Test ROC-AUC | **0.9421** |
| Test Average Precision | **0.8880** |

The final trained pipeline is saved as:

`model/model.pkl`

Training evaluation metadata is saved as:

`model/model_meta.json`

These results are from the recorded held-out test evaluation. They should not be interpreted as a guarantee of future lending performance.

## 5. Project Structure

```text
loan-default-project/
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── loan_dataset.csv
│
├── model/
│   ├── model.pkl
│   └── model_meta.json
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
│
├── scripts/
│   └── test_api.py
│
├── src/
│   └── train.py
│
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── fly.toml
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

## 6. Installation and Reproducibility

### Requirements

- Python 3.11
- uv package manager
- Git

Clone the repository:

```bash
git clone https://github.com/David-Owusu-Nana-Yaw-Bososmpem/loan-default-project.git
cd loan-default-project
```

Install uv if necessary:

Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment and install dependencies:

```bash
uv sync --default-index https://pypi.org/simple
```

The lockfile specifies the dependency versions used for reproducible installation.

## 7. Train the Model

Run:

```bash
uv run python src/train.py
```

The training script:

1. Loads and cleans the dataset.
2. Engineers financial features.
3. Splits the dataset into training and testing sets.
4. Trains and compares four classification models.
5. Tunes the selected model using RandomizedSearchCV.
6. Evaluates the final model.
7. Saves the trained pipeline and evaluation metadata.

The generated files are:

```text
model/
├── model.pkl
└── model_meta.json
```

## 8. Run the API Locally

Start the FastAPI application:

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Available local endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Application and model health |
| `/predict` | POST | Predict loan default probability |
| `/docs` | GET | Interactive Swagger documentation |

Open:

http://127.0.0.1:8000/docs

### Example prediction request

```json
{
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
}
```

### Example prediction response

```json
{
  "default_probability": 0.0868,
  "prediction": "NO DEFAULT",
  "risk_level": "LOW"
}
```

The API uses Pydantic for request and response validation.

Risk classification follows these thresholds:

| Default probability | Risk level |
|---|---|
| Below 30% | LOW |
| 30% to below 70% | MEDIUM |
| 70% and above | HIGH |

The binary prediction threshold is 50%.

## 9. API Testing

Run the API test script while the local server is running:

```bash
uv run python scripts/test_api.py
```

The test script checks:

- Request validation.
- Successful HTTP prediction requests.
- Response validation.
- Invalid loan grade rejection.

## 10. Docker Deployment

Build the Docker image:

```bash
docker build -t loan-default-api .
```

Run the container:

```bash
docker run -p 8000:8000 loan-default-api
```

Open:

http://localhost:8000/docs

The Docker image contains the FastAPI application, trained model, and required production dependencies.

## 11. Cloud Deployment with Fly.io

The application is deployed on Fly.io using Docker.

**Application URL:**

https://loan-default-project.fly.dev

**Swagger documentation:**

https://loan-default-project.fly.dev/docs

**Health endpoint:**

https://loan-default-project.fly.dev/health

The deployed application uses:

- Docker-based deployment.
- FastAPI and Uvicorn.
- Python 3.11.
- A serialized Scikit-learn model.
- Fly.io Machines.

To deploy future updates using the Fly CLI:

```bash
fly auth login
fly deploy
```

## 12. Cloud Deployment Verification

The deployed API was tested successfully.

### Health check

GET `/health`

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Prediction test

POST `/predict`

For the example loan application, the deployed API returned:

```json
{
  "default_probability": 0.0868,
  "prediction": "NO DEFAULT",
  "risk_level": "LOW"
}
```

The cloud prediction matched the local API prediction for the same input.

HTTP response: `200 OK`.

## 13. Technologies Used

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Uvicorn
- uv
- Docker
- Fly.io
- Git and GitHub
- Jupyter Notebook

## 14. Limitations

The model estimates default probability based on patterns in historical loan applications.

Its predictions should be treated as decision-support outputs rather than automatic approval or rejection decisions.

Further work could include:

- Probability calibration.
- Fairness and bias assessment.
- Model explainability using SHAP.
- Monitoring model drift.
- Testing on newer and independent datasets.

## 15. Evaluation Criteria Addressed

- Defined a practical machine learning problem.
- Conducted exploratory data analysis.
- Performed data cleaning and feature engineering.
- Compared four classification algorithms.
- Tuned the selected model.
- Evaluated model performance using ROC-AUC and Average Precision.
- Saved the final trained model.
- Developed a REST API with FastAPI.
- Implemented Pydantic validation.
- Tested the API locally.
- Containerized the application with Docker.
- Deployed the API to Fly.io.
- Verified live cloud predictions.

## Author

David Owusu Bosompem

Machine Learning Zoomcamp Midterm Project

**GitHub:** https://github.com/David-Owusu-Nana-Yaw-Bososmpem

**Live API:** https://loan-default-project.fly.dev

**API Documentation:** https://loan-default-project.fly.dev/docs

---

Educational project developed as part of the Machine Learning Zoomcamp midterm assessment.
