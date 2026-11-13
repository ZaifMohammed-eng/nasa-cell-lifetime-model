# One image for the Kubeflow pipeline steps, the FastAPI service and the
# in-cluster MLflow server. Not built where this was written (no Docker daemon);
# build and smoke-test it before pushing (see DEPLOY.md).
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY mlops ./mlops
COPY serve ./serve
COPY pipelines ./pipelines
COPY data ./data
# Editable install keeps the package rooted at /app, so mlops.train / mlops.monitor
# find /app/data and write to /app/results.
RUN pip install --no-cache-dir -e ".[serve,mlops]" && mkdir -p results logs
ENV MODEL_URI="models:/cell-eol-forecaster@champion" \
    MLFLOW_TRACKING_URI="http://mlflow.kubeflow.svc.cluster.local:5000" \
    REQUEST_LOG="/app/logs/requests.jsonl" \
    MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
