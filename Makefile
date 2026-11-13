PY ?= python
MLFLOW_TRACKING_URI ?= sqlite:///mlruns.db
export MLFLOW_TRACKING_URI
export MLFLOW_DISABLE_AGENT_HINT=1

install:   ; $(PY) -m pip install -e ".[mlops,serve,dev]"
test:      ; $(PY) -m pytest -q
train:     ; $(PY) -m mlops.train --trials 60 --tracking-uri $(MLFLOW_TRACKING_URI)
monitor:   ; $(PY) -m mlops.monitor --tracking-uri $(MLFLOW_TRACKING_URI)
pipeline:  ; $(PY) pipelines/kfp_pipeline.py --compile pipelines/cell_eol_pipeline.yaml
pipeline-local: ; $(PY) pipelines/kfp_pipeline.py --local --trials 60
serve:     ; uvicorn serve.app:app --port 8000
bench:     ; $(PY) -m mlops.bench --n 2000 --concurrency 16 --shadow ""
mlflow-server: ; mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
