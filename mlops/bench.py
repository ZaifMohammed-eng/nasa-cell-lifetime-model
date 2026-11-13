"""Load test for the running service: starts uvicorn, replays real cell histories.

Usage: python -m mlops.bench [--n 2000] [--concurrency 16] [--shadow models:/cell-eol-forecaster/1]
"""
from __future__ import annotations

import argparse, asyncio, json, os, platform, subprocess, sys, time

import httpx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def payloads(n, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.concat([pd.read_csv(os.path.join(ROOT, "data", f)) for f in ("capacity.csv", "shift_capacity.csv")])
    cells = sorted(df.cell.unique())
    out = []
    for _ in range(n):
        c = cells[rng.integers(len(cells))]
        g = df[df.cell == c].sort_values("cycle")
        k = int(rng.integers(10, min(len(g), 120) + 1))
        rows = g.iloc[:k][["cycle", "capacity_Ah", "rest_before_h", "T_max_C"]]
        rows = rows.astype(object).where(rows.notna(), None)
        out.append({"cell_id": c, "cycles": rows.to_dict("records")})
    return out


async def run(url, reqs, conc):
    lat, codes = [], []
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=30) as cl:
        async def one(p):
            async with sem:
                t = time.perf_counter()
                r = await cl.post(url, json=p)
                lat.append((time.perf_counter() - t) * 1000)
                codes.append(r.status_code)
        t0 = time.perf_counter()
        await asyncio.gather(*(one(p) for p in reqs))
        wall = time.perf_counter() - t0
    return np.array(lat), codes, wall


def main(n, conc, shadow, port=8765, workers=1):
    env = {**os.environ, "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlruns.db"),
           "REQUEST_LOG": os.path.join(ROOT, "logs", "bench_requests.jsonl"), "MLFLOW_DISABLE_AGENT_HINT": "1"}
    if shadow:
        env["SHADOW_MODEL_URI"] = shadow
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "serve.app:app", "--port", str(port),
                             "--workers", str(workers), "--log-level", "warning"], cwd=ROOT, env=env)
    try:
        for _ in range(120):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        health = httpx.get(f"http://127.0.0.1:{port}/health").json()
        reqs = payloads(n)
        asyncio.run(run(f"http://127.0.0.1:{port}/v1/forecast", reqs[:50], 4))  # warm-up
        res = {"health": health, "n": n, "workers": workers, "cpu_count": os.cpu_count(),
               "platform": platform.platform(), "python": platform.python_version()}
        for label, c in (("sequential", 1), (f"concurrency_{conc}", conc)):
            lat, codes, wall = asyncio.run(run(f"http://127.0.0.1:{port}/v1/forecast", reqs, c))
            res[label] = {"p50_ms": round(float(np.percentile(lat, 50)), 2),
                          "p95_ms": round(float(np.percentile(lat, 95)), 2),
                          "p99_ms": round(float(np.percentile(lat, 99)), 2),
                          "throughput_rps": round(n / wall, 1),
                          "non_200": int(sum(x != 200 for x in codes))}
        m = httpx.get(f"http://127.0.0.1:{port}/metrics").text
        res["decision_counts"] = {l.split('"')[1]: float(l.split()[-1]) for l in m.splitlines()
                                  if l.startswith("eol_decisions_total{")}
    finally:
        proc.terminate(); proc.wait()
    out = os.path.join(ROOT, "results", "latency_benchmark.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--shadow", default="models:/cell-eol-forecaster/1")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    main(a.n, a.concurrency, a.shadow, workers=a.workers)
