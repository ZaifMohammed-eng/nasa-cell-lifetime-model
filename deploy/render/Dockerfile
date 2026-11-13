# Public demo on Render (free web service). Build context = repo root.
# Serves the exported champion (serve/model/champion.json): no MLflow at runtime.
FROM python:3.12-slim
WORKDIR /app
COPY deploy/space/requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt
COPY src/__init__.py src/lifetime.py ./src/
COPY mlops/__init__.py mlops/core.py ./mlops/
COPY serve/__init__.py serve/app.py serve/cases.py ./serve/
COPY serve/static ./serve/static
COPY serve/model ./serve/model
COPY data/capacity.csv data/shift_capacity.csv ./data/
ENV PYTHONPATH=/app \
    MODEL_URI=/app/serve/model/champion.json \
    REQUEST_LOG=/tmp/requests.jsonl \
    RATE_LIMIT_PER_MIN=120
# Render sets $PORT (default 10000) and sends client IPs in X-Forwarded-For.
CMD ["sh", "-c", "uvicorn serve.app:app --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers --forwarded-allow-ips '*'"]
