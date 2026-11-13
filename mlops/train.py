"""Tune, evaluate, register.

1. Baselines: fleet-mean EOL and the untuned research configuration.
2. Nested leave-one-cell-out: for each outer cell, Optuna tunes HParams with an
   inner LOCO over the other three cells (objective = 90 % interval score, a
   proper scoring rule that charges both width and misses); the tuned model is
   then scored once on the outer cell. This is the number used for promotion.
3. Final tuning on all cells, fit, log to MLflow as a pyfunc model, register
   it, and set alias "challenger". Promote to "champion" only if the nested
   interval score beats the current champion's.

Usage: python -m mlops.train [--trials 40] [--tracking-uri sqlite:///mlruns.db]
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, tempfile
from dataclasses import asdict, replace

import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from mlops.core import CellEOLForecaster, HParams, fleet_mean_baseline, loco_eval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "cell-eol-forecaster"
KS = (30, 40, 50)


def suggest(trial) -> HParams:
    return HParams(
        prior_scale=trial.suggest_float("prior_scale", 1.0, 32.0, log=True),
        long_rest_h=trial.suggest_categorical("long_rest_h", [6.0, 8.0, 12.0, 18.0, 24.0]),
        noise_mult=trial.suggest_float("noise_mult", 0.5, 16.0, log=True),
        cov_scale=trial.suggest_float("cov_scale", 0.01, 20.0, log=True),
    )


def tune(df, cells, trials, seed=0) -> tuple[HParams, float]:
    def obj(trial):
        return loco_eval(df, suggest(trial), KS, cells)["interval_score"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.enqueue_trial({"prior_scale": 4.0, "long_rest_h": 12.0, "noise_mult": 1.0, "cov_scale": 1.0})
    st.optimize(obj, n_trials=trials)
    return replace(HParams(), **st.best_params), st.best_value


class PyfuncForecaster(mlflow.pyfunc.PythonModel):
    """Input: one row per cycle (cell, cycle, capacity_Ah, rest_before_h, T_max_C). Output: one row per cell."""

    def load_context(self, context):
        from mlops.core import CellEOLForecaster
        with open(context.artifacts["model_json"]) as fh:
            self.m = CellEOLForecaster.from_json(fh.read())

    def predict(self, context, model_input: pd.DataFrame, params=None):
        out = []
        for cell, g in model_input.groupby("cell", sort=False):
            p = self.m.predict(g)
            out.append({"cell": cell, "decision": p["decision"], "eol_p05": p["eol_p05"],
                        "eol_p50": p["eol_p50"], "eol_p95": p["eol_p95"], "reasons": ";".join(p["reasons"])})
        return pd.DataFrame(out)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(trials: int, tracking_uri: str, out_json: str | None = None):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("cell-eol-forecaster")
    data_path = os.path.join(ROOT, "data", "capacity.csv")
    df = pd.read_csv(data_path)
    cells = sorted(df.cell.unique())
    data_sha = hashlib.sha256(open(data_path, "rb").read()).hexdigest()[:12]
    summary = {}

    with mlflow.start_run(run_name="tune-evaluate-register") as run:
        mlflow.set_tags({"git_sha": git_sha(), "data_sha256": data_sha, "cells": ",".join(cells)})
        mlflow.log_artifact(data_path, "data")

        # ---- baselines
        fleet = fleet_mean_baseline(df, KS)
        default = loco_eval(df, HParams(), KS)
        mlflow.log_metrics({"baseline_fleet_mean_mae": fleet, "default_mae": default["mae"],
                            "default_coverage90": default["coverage90"],
                            "default_interval_score": default["interval_score"],
                            "default_mean_width": default["mean_width"]})
        summary["baseline"] = {"fleet_mean_mae": fleet, **{k: v for k, v in default.items() if k != "rows"}}

        # ---- nested LOCO
        outer_rows, outer_default_rows = [], []
        for c in cells:
            inner = [o for o in cells if o != c]
            with mlflow.start_run(run_name=f"outer-{c}", nested=True):
                hp, inner_best = tune(df, inner, trials)
                mlflow.log_params(asdict(hp)); mlflow.log_metric("inner_interval_score", inner_best)
                for tag, h in (("tuned", hp), ("default", HParams())):
                    m = CellEOLForecaster.fit(df[df.cell.isin(inner)], h)
                    d = df[df.cell == c].sort_values("cycle")
                    from src.lifetime import true_eol
                    te = true_eol(d.capacity_Ah.values, float(d.capacity_Ah.values[:3].mean()))
                    for k in KS:
                        p = m.predict(d.iloc[:k])
                        row = dict(cell=c, k=k, true_eol=te, p05=p["eol_p05"], p50=p["eol_p50"], p95=p["eol_p95"],
                                   decision=p["decision"], **{f"hp_{a}": b for a, b in asdict(h).items()})
                        (outer_rows if tag == "tuned" else outer_default_rows).append(row)
        def score(rows):
            r = pd.DataFrame(rows)
            r["abs_err"] = (r.p50 - r.true_eol).abs()
            r["covered"] = (r.p05 <= r.true_eol) & (r.true_eol <= r.p95)
            lo, hi, y = r.p05, r.p95, r.true_eol
            r["iscore"] = (hi - lo) + 20 * (lo - y).clip(lower=0) + 20 * (y - hi).clip(lower=0)
            r["width"] = hi - lo
            return r, {"mae": float(r.abs_err.mean()), "coverage90": float(r.covered.mean()),
                       "interval_score": float(r.iscore.mean()), "mean_width": float(r.width.mean()),
                       "forecast_rate": float((r.decision == "forecast").mean())}
        rt, st = score(outer_rows)
        rd, sd = score(outer_default_rows)
        mlflow.log_metrics({f"nested_tuned_{k}": v for k, v in st.items()})
        mlflow.log_metrics({f"nested_default_{k}": v for k, v in sd.items()})
        with tempfile.TemporaryDirectory() as td:
            rt.to_csv(os.path.join(td, "nested_tuned.csv"), index=False)
            rd.to_csv(os.path.join(td, "nested_default.csv"), index=False)
            mlflow.log_artifacts(td, "evaluation")
        summary["nested"] = {"tuned": st, "default": sd}

        # ---- final model
        hp, best = tune(df, cells, trials)
        model = CellEOLForecaster.fit(df, hp)
        final_eval = loco_eval(df, hp, KS)
        mlflow.log_params({f"final_{k}": v for k, v in asdict(hp).items()})
        mlflow.log_metrics({"final_loco_interval_score": final_eval["interval_score"],
                            "final_loco_coverage90": final_eval["coverage90"], "final_loco_mae": final_eval["mae"]})
        summary["final"] = {"hp": asdict(hp), **{k: v for k, v in final_eval.items() if k != "rows"}}

        with tempfile.TemporaryDirectory() as td:
            mj = os.path.join(td, "model.json")
            open(mj, "w").write(model.to_json())
            example = df[df.cell == "B0005"].sort_values("cycle").iloc[:40][
                ["cell", "cycle", "capacity_Ah", "rest_before_h", "T_max_C"]]
            pm = PyfuncForecaster()
            pm.m = model
            sig = infer_signature(example, pm.predict(None, example))
            info = mlflow.pyfunc.log_model(name="model", python_model=pm, artifacts={"model_json": mj},
                                           signature=sig, input_example=example,
                                           code_paths=[os.path.join(ROOT, "mlops"), os.path.join(ROOT, "src")],
                                           registered_model_name=MODEL_NAME)

    # ---- registry: challenger -> champion gate
    client = MlflowClient()
    v = info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "challenger", v)
    client.set_model_version_tag(MODEL_NAME, v, "nested_interval_score", f"{st['interval_score']:.2f}")
    client.set_model_version_tag(MODEL_NAME, v, "git_sha", git_sha())
    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champ_score = float(champ.tags.get("nested_interval_score", "inf"))
    except Exception:
        champ, champ_score = None, float("inf")
    promote = st["interval_score"] < champ_score
    if promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", v)
    summary["registry"] = {"version": v, "promoted": promote,
                           "previous_champion": None if champ is None else champ.version,
                           "run_id": run.info.run_id, "git_sha": git_sha(), "data_sha256": data_sha}
    print(json.dumps(summary, indent=1, default=float))
    if out_json:
        json.dump(summary, open(out_json, "w"), indent=1, default=float)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlruns.db"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mlops_metrics.json"))
    a = ap.parse_args()
    main(a.trials, a.tracking_uri, a.out)
