# Hugging Face Docker Space: public, always-on demo (free CPU tier). Serves the
# exported champion from JSON, so it needs no MLflow server at runtime.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt
COPY --chown=user . .
USER user
ENV PYTHONPATH=/app \
    MODEL_URI=/app/serve/model/champion.json \
    REQUEST_LOG=/tmp/requests.jsonl \
    RATE_LIMIT_PER_MIN=120
EXPOSE 7860
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
