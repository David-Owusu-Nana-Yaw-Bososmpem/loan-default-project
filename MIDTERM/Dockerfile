FROM python:3.11-slim

WORKDIR /app

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock requirements.txt ./

# Install dependencies from lock-exported requirements
RUN uv pip install --system --no-cache -r requirements.txt

COPY app/ ./app/
COPY model/ ./model/
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
