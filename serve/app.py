"""Real-time EOL forecast service.

* Loads the model registered under alias `champion` from the MLflow registry
  once at startup (MODEL_URI overrides, e.g. a local model directory).
* Optional shadow model (SHADOW_MODEL_URI, e.g. models:/cell-eol-forecaster@challenger):
  scored on every request in a background task after the response is sent,
  logged, never returned to the caller.
* Every response carries a decision: forecast | route_to_engineer | reject |
  eol_already_reached, plus reason codes. Out-of-envelope inputs, short
  histories and wide intervals are routed to a person instead of answered.
* Prometheus metrics at /metrics; one JSON line per request in REQUEST_LOG for
  drift monitoring (mlops/monitor.py) and for joining against observed EOL later.
"""
from __future__ import annotations

import json, os, time, uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_URI = os.environ.get("MODEL_URI", "models:/cell-eol-forecaster@champion")
SHADOW_URI = os.environ.get("SHADOW_MODEL_URI", "")
REQUEST_LOG = os.environ.get("REQUEST_LOG", "logs/requests.jsonl")
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))   # per client IP; 0 disables
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

LAT = Histogram("eol_request_latency_seconds", "Forecast latency", buckets=(.001, .0025, .005, .01, .025, .05, .1, .25))
DEC = Counter("eol_decisions_total", "Decisions returned", ["decision"])
SHADOW_DIFF = Histogram("eol_shadow_abs_diff_cycles", "|champion p50 - shadow p50|", buckets=(1, 2, 5, 10, 20, 50, 100))


class Cycle(BaseModel):
    cycle: int = Field(ge=1)
    capacity_Ah: float
    rest_before_h: float | None = None
    T_max_C: float | None = None


class ForecastRequest(BaseModel):
    cell_id: str
    cycles: list[Cycle] = Field(min_length=1, max_length=5000)


class ForecastResponse(BaseModel):
    request_id: str
    cell_id: str
    decision: Literal["forecast", "route_to_engineer", "reject", "eol_already_reached"]
    eol_p05: float | None
    eol_p50: float | None
    eol_p95: float | None
    remaining_cycles_p50: float | None
    reasons: list[str]
    model_version: str
    latency_ms: float


def _load(uri: str):
    if uri.endswith(".json"):
        from mlops.core import CellEOLForecaster
        meta_path = uri.replace(".json", ".meta.json")
        STATE["meta"] = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        label = f"{STATE['meta']['model']}/v{STATE['meta']['version']}" if STATE["meta"] else os.path.basename(uri)
        return CellEOLForecaster.from_json(open(uri).read()), label
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
    pm = mlflow.pyfunc.load_model(uri)
    version = uri
    if uri.startswith("models:/") and "@" in uri:
        name, alias = uri[len("models:/"):].split("@")
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(name, alias)
        version = f"{name}/v{mv.version}"
        if "meta" not in STATE:
            mets = client.get_run(mv.run_id).data.metrics
            STATE["meta"] = {"model": name, "alias": alias, "version": mv.version, "run_id": mv.run_id,
                             "nested_loco": {k.replace("nested_", ""): v for k, v in mets.items() if k.startswith("nested_")},
                             "baseline_fleet_mean_mae": mets.get("baseline_fleet_mean_mae")}
    return pm.unwrap_python_model().m, version


STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.clear()
    RATE.clear()
    STATE["model"], STATE["version"] = _load(os.environ.get("MODEL_URI", MODEL_URI))
    shadow = os.environ.get("SHADOW_MODEL_URI", SHADOW_URI)
    if shadow:
        STATE["shadow"], STATE["shadow_version"] = _load(shadow)
    log = os.environ.get("REQUEST_LOG", REQUEST_LOG)
    os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
    STATE["log"] = open(log, "a", buffering=1)
    yield
    STATE["log"].close()


app = FastAPI(title="Cell EOL forecaster", version="1.0.0", lifespan=lifespan,
              description="Forecasts cycles to 80 % capacity for an 18650 Li-ion cell from its early cycle "
                          "history, with a 90 % interval, and routes inputs outside the training envelope to "
                          "an engineer. Demo page at `/`; built-in test cases at `/v1/cases`.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

RATE: dict = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Sliding one-minute window per client IP (first X-Forwarded-For hop behind a proxy)."""
    if RATE_LIMIT_PER_MIN > 0 and request.method == "POST":
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
        now, q = time.time(), RATE[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            return JSONResponse({"detail": f"rate limit: {RATE_LIMIT_PER_MIN} POST requests per minute"},
                                status_code=429, headers={"Retry-After": "60"})
        q.append(now)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model_version": STATE.get("version"), "shadow": STATE.get("shadow_version")}


@app.get("/v1/model")
def model_info():
    m = STATE["model"]
    return {"version": STATE["version"], "hparams": m.hp.__dict__, "envelope": m.envelope,
            "train_cells": m.train_cells, "registry": STATE.get("meta", {})}


@app.get("/v1/cases")
def list_cases():
    from serve.cases import catalog
    return [{k: v for k, v in c.items() if k not in ("source", "edit")} for c in catalog()]


@app.post("/v1/cases/{case_id}/run")
def run_case(case_id: str):
    from serve.cases import run
    try:
        return run(case_id, STATE["model"])
    except KeyError:
        raise HTTPException(404, f"unknown case {case_id}")


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest, bg: BackgroundTasks):
    t0 = time.perf_counter()
    hist = pd.DataFrame([c.model_dump() for c in req.cycles]).astype({"rest_before_h": float, "T_max_C": float})
    p = STATE["model"].predict(hist)
    ms = (time.perf_counter() - t0) * 1000
    LAT.observe(ms / 1000)
    DEC.labels(p["decision"]).inc()
    rid = uuid.uuid4().hex[:12]
    rec = {"ts": time.time(), "request_id": rid, "cell_id": req.cell_id, "model": STATE["version"],
           "n_cycles": len(hist), "q0_Ah": float(hist.capacity_Ah.iloc[:3].mean()),
           "T_max_median": float(hist.T_max_C.median()) if hist.T_max_C.notna().any() else None,
           "decision": p["decision"], "eol_p50": p["eol_p50"], "reasons": p["reasons"], "latency_ms": ms}
    # Shadow scoring and logging run after the response is sent, off the latency path.
    bg.add_task(_shadow_and_log, hist, p, rec)
    rem = None if p["eol_p50"] is None else max(p["eol_p50"] - int(hist.cycle.max()), 0.0)
    return ForecastResponse(request_id=rid, cell_id=req.cell_id, decision=p["decision"], eol_p05=p["eol_p05"],
                            eol_p50=p["eol_p50"], eol_p95=p["eol_p95"], remaining_cycles_p50=rem,
                            reasons=p["reasons"], model_version=STATE["version"], latency_ms=round(ms, 3))


def _shadow_and_log(hist, p, rec):
    if "shadow" in STATE:
        s = STATE["shadow"].predict(hist)
        rec["shadow"] = {"version": STATE["shadow_version"], "decision": s["decision"], "eol_p50": s["eol_p50"]}
        if s["eol_p50"] is not None and p["eol_p50"] is not None:
            SHADOW_DIFF.observe(abs(s["eol_p50"] - p["eol_p50"]))
    STATE["log"].write(json.dumps(rec) + "\n")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
