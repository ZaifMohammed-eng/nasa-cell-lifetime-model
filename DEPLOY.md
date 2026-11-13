# Deploying the cell EOL forecaster (no cluster needed)

| Goal | Section |
|---|---|
| Put the code on GitHub | 1 |
| Run MLflow + the Kubeflow pipeline + the API on your Mac with one command | 2 |
| Let anyone test it while your Mac is on (temporary link) | 3 |
| Public link that stays up without your Mac (free) | 4 |
| Same thing without Docker | 5 |
| Kubernetes, if you ever have a cluster | 6 |

What was tested before release (Linux, no Docker daemon available):
the three compose services run as plain processes with the exact compose
commands (MLflow 3 server with proxied artifacts, Kubeflow pipeline on the KFP
local runner, API loading `@champion` from the registry, 22/22 demo cases), the
Hugging Face bundle in a clean virtualenv without MLflow (22/22 cases, rate
limit returns 429 per IP), and the demo page in headless Chrome. **Not**
tested: `docker build`, `docker compose` itself, the tunnel and the Space build.
Their first run is yours; send any error output back.

## 0. Tools (macOS, Apple Silicon)

```bash
brew install git
# Docker Desktop: https://www.docker.com/products/docker-desktop/  (Settings -> Resources: 4 CPUs, 8 GB)
```

## 1. Git: put the branch on GitHub

```bash
cd ~/path/to/nasa-cell-lifetime-model     # your existing clone (or: git clone https://github.com/ZaifMohammed-eng/nasa-cell-lifetime-model.git)
git checkout main && git pull
git checkout -b mlops-production
unzip -q ~/Downloads/nasa-cell-lifetime-model-mlops.zip -d /tmp/mlops
rsync -a /tmp/mlops/nasa-cell-lifetime-model/ ./   # zip has one top-level folder
git status                                          # new: mlops/ serve/ pipelines/ deploy/ tests/ ...
git add -A
git commit -m "Add production layer: MLflow registry, Kubeflow pipeline, FastAPI service, shift gates, CI"
git push -u origin mlops-production
```

Then on GitHub:
1. **Actions** tab: the `ci` workflow runs tests, compiles the pipeline, does a
   smoke train and runs the shift gate. Wait for green.
2. Open a pull request `mlops-production → main` and merge it, so the resume's
   `[repo]` link shows the production layer on the default branch.

## 2. Local deployment: one command (Docker Compose)

```bash
cd ~/path/to/nasa-cell-lifetime-model
docker compose up --build
```

What starts, in order:

| Service | What it does | Where |
|---|---|---|
| `mlflow` | Tracking server + model registry (SQLite + artifacts in a Docker volume) | http://localhost:5001 |
| `pipeline` | The Kubeflow pipeline on the KFP local runner: data check → Optuna tuning with nested LOCO → register in MLflow → shift check → release note; then the drift report. Exits when done (~1–2 min) | logs in the terminal |
| `api` | FastAPI service, loads `models:/cell-eol-forecaster@champion` from MLflow; demo page with 22 test cases | http://localhost:8000 |

Check it:
- http://localhost:8000: demo page. Click a case, or **Run all**; expect 22/22.
- http://localhost:8000/docs: interactive API docs.
- http://localhost:5001: MLflow. Experiment `cell-eol-forecaster` (nested runs,
  metrics, evaluation CSVs) and **Models → cell-eol-forecaster** with `@champion`.

Every `docker compose up` retrains. The first run promotes v1; later runs on the
same data register v2, v3… and the release note says "held back". Stop with
`Ctrl-C`; `docker compose down -v` also deletes the MLflow volume (registry reset).

## 3. Share it from your Mac (temporary public link)

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8000      # prints https://<random>.trycloudflare.com
caffeinate -dimsu                                    # second terminal: stop the Mac sleeping
```

Send the `trycloudflare.com` link. It works only while both commands and
`docker compose up` keep running, and the URL changes every time. Tunnel
**only port 8000**: the MLflow UI on 5001 has no login and must stay local.
The API limits each visitor IP to 120 POST requests per minute (`RATE_LIMIT_PER_MIN`).

## 4. Public link that stays up: Hugging Face Space (free)

The Space serves the exported champion from a JSON file, so it needs no MLflow
or pipeline at runtime.

1. Create an account at https://huggingface.co, then **New Space**:
   name `cell-eol-forecaster`, SDK **Docker**, template **Blank**, hardware
   **CPU basic (free)**, visibility **Public**.
2. **Settings → Access Tokens → New token** with *write* access.
3. Build and push:

```bash
cd ~/path/to/nasa-cell-lifetime-model
pip install -e ".[mlops,serve,dev]"                 # only needed if you re-export the model
# serve/model/champion.json is already in the repo. To ship a model you trained
# in section 2 instead:  MLFLOW_TRACKING_URI=http://localhost:5001 python -m mlops.export
git clone https://huggingface.co/spaces/<your-hf-username>/cell-eol-forecaster ~/hf-space
bash deploy/space/build_space.sh ~/hf-space
cd ~/hf-space && git add -A && git commit -m "Deploy cell EOL forecaster demo" && git push
# username = your HF username, password = the write token
```

4. The **Logs** tab shows the build (~2–4 min). The app is then live at
   `https://<your-hf-username>-cell-eol-forecaster.hf.space`, and the Space page is
   `https://huggingface.co/spaces/<your-hf-username>/cell-eol-forecaster`.
5. Free Spaces sleep after a period without visitors; the first visit after
   that takes about a minute to wake. To update, rerun `build_space.sh` and push.

Put that link on your resume and in the GitHub README once it loads.

## 5. Without Docker

```bash
conda create -n mlops python=3.12 -y && conda activate mlops
pip install -e ".[mlops,serve,dev]"
pytest -q                                  # 21 tests

# Terminal 1: tracking server + registry (port 5001: macOS AirPlay Receiver holds 5000)
make mlflow-server                         # = mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001

# Terminal 2
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
make train        # Optuna + nested LOCO, registers v1, sets alias champion (~30 s)
make train        # identical retrain: registered as v2, NOT promoted (gate demo)
make monitor      # replays 5 unseen-protocol NASA cells -> results/drift_report.json
```

Open http://127.0.0.1:5001: **Experiments → cell-eol-forecaster** shows the
parent run, the four `outer-*` nested runs, metrics and evaluation CSVs.
**Models → cell-eol-forecaster** shows the versions and the `@champion` alias.
Expected: nested MAE 12.7 cycles, interval score 66.1, coverage 0.75.

Serve the champion locally:

```bash
make serve                                  # demo page http://localhost:8000, API docs /docs
curl -s localhost:8000/health
python - <<'EOF' > /tmp/req.json
import json, pandas as pd
g = pd.read_csv("data/capacity.csv").query("cell == 'B0006'").sort_values("cycle").iloc[:40]
g = g[["cycle","capacity_Ah","rest_before_h","T_max_C"]].astype(object).where(g.notna(), None)
print(json.dumps({"cell_id": "B0006", "cycles": g.to_dict("records")}))
EOF
curl -s -X POST localhost:8000/v1/forecast -H 'Content-Type: application/json' -d @/tmp/req.json
make bench                                  # load test -> results/latency_benchmark.json
```

Kubeflow pipeline without a cluster:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
make pipeline-local     # data_check -> tune_evaluate_register -> shift_check -> release_note
```

## 6. Kubernetes (only if you have a cluster)

### 6a. kind on your Mac

```bash
kind create cluster --name mlops

# Kubeflow Pipelines standalone. Use the latest tag from
# https://github.com/kubeflow/pipelines/releases (docs currently show 2.17.0 / 2.18.0).
export PIPELINE_VERSION=2.17.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
kubectl -n kubeflow get pods -w         # ~3-5 min until ml-pipeline* pods are Running
# If pods keep crashing on kind, use env/platform-agnostic instead of env/dev.
# proxy-agent crash-looping is expected outside GCP and harmless.

# One image for pipeline steps, the API and the in-cluster MLflow
docker build -t ghcr.io/zaifmohammed-eng/cell-eol-forecaster:1.0.0 .
kind load docker-image ghcr.io/zaifmohammed-eng/cell-eol-forecaster:1.0.0 --name mlops

# MLflow inside the cluster
kubectl apply -f deploy/k8s/mlflow.yaml
kubectl -n kubeflow rollout status deploy/mlflow

# Terminal A / B: UIs
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80     # Kubeflow UI  -> http://localhost:8080
kubectl port-forward -n kubeflow svc/mlflow 5002:5000           # MLflow UI    -> http://localhost:5002

# Submit the compiled pipeline (pipelines/cell_eol_pipeline.yaml)
python pipelines/submit.py --host http://localhost:8080
```

Watch the run in the Kubeflow UI (**Runs**): four steps; `release-note` says
whether the new version was promoted or held back. The model appears in MLflow at localhost:5002.

Deploy the API from the registry:

```bash
kubectl apply -f deploy/k8s/serve.yaml
kubectl -n kubeflow rollout status deploy/eol-forecaster
kubectl port-forward -n kubeflow svc/eol-forecaster 8000:8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/forecast -H 'Content-Type: application/json' -d @/tmp/req.json
# After a new champion is promoted:
kubectl -n kubeflow rollout restart deploy/eol-forecaster
```

Clean up: `kind delete cluster --name mlops`.

### 6b. Cloud cluster (GKE/EKS)

Push the image instead of `kind load`. Cloud nodes are amd64, your Mac is arm64:

```bash
echo $GITHUB_PAT | docker login ghcr.io -u ZaifMohammed-eng --password-stdin   # PAT with write:packages
docker buildx build --platform linux/amd64 -t ghcr.io/zaifmohammed-eng/cell-eol-forecaster:1.0.0 --push .
```

Make the GHCR package public, or add an `imagePullSecret`. For more than one
user, replace SQLite with Postgres and the PVC with S3/MinIO in `deploy/k8s/mlflow.yaml`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `api` never starts in compose | It waits for `pipeline` to finish successfully: read the pipeline logs with `docker compose logs pipeline` |
| Demo page shows "Error: rate limit" | More than `RATE_LIMIT_PER_MIN` POSTs from one IP in a minute; wait 60 s |
| Space build fails | Check the Logs tab; the bundle needs `serve/model/champion.json` (run `python -m mlops.export`) |
| MLflow returns **403** to pods | MLflow 3 checks the Host header. Add the service name to `--allowed-hosts` (already set in `deploy/k8s/mlflow.yaml`) |
| `make mlflow-server` fails, address in use | macOS AirPlay Receiver on port 5000/7000; the Makefile uses 5001 |
| `ImagePullBackOff` on kind | Image tag mismatch, or `kind load docker-image` was skipped |
| API pod crash-loops at start-up (model load fails) | Run the pipeline (or `make train` against that MLflow) before deploying `serve.yaml` |
| Pipeline step finishes instantly | KFP caching; `submit.py` disables it, so check runs created from the UI |
| Steps can't import `mlops` | Pipeline compiled with a different image name; set `PIPELINE_IMAGE` and run `make pipeline` |
