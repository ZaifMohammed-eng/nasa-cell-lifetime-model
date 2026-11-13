"""Distribution-shift check against the training data.

Replays cells the model never saw through the champion and reports, per cell,
the decision and reason codes, plus a population stability index (PSI) for
per-cycle inputs (discharge peak temperature, end voltage, rest time, capacity
relative to initial), computed over the same first-k-cycle window as the
reference. The shift cells are other NASA PCoE groups run at 4 C,
43 C, or a 4 A discharge, i.e. protocols the model was not trained on.

Usage: python -m mlops.monitor [--model-uri models:/cell-eol-forecaster@champion]
"""
from __future__ import annotations

import argparse, json, os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["T_max_C", "V_end", "rest_before_h", "cap_rel"]
PSI_ALERT = 0.25


def psi(ref, cur, bins=10):
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def add_rel(df):
    df = df.sort_values(["cell", "cycle"]).copy()
    df["cap_rel"] = df.capacity_Ah / df.groupby("cell").capacity_Ah.transform(lambda s: s.iloc[:3].mean())
    return df


def main(model_uri, tracking_uri, k=40):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(model_uri).unwrap_python_model().m
    ref = add_rel(pd.read_csv(os.path.join(ROOT, "data", "capacity.csv")))
    # Reference = the same early-life window of the training cells. Comparing a
    # cell's first k cycles with whole-life data flags every cell, shifted or not.
    ref = ref[ref.cycle <= k]
    shift = add_rel(pd.read_csv(os.path.join(ROOT, "data", "shift_capacity.csv")))
    report = {"model_uri": model_uri, "k": k, "psi_alert": PSI_ALERT, "cells": {}}
    for c, g in shift.groupby("cell"):
        g = g.sort_values("cycle").iloc[:k]
        p = model.predict(g)
        report["cells"][c] = {
            "decision": p["decision"], "reasons": p["reasons"],
            "psi": {f: round(psi(ref[f], g[f]), 2) for f in FEATURES},
        }
        report["cells"][c]["psi_alerts"] = [f for f, v in report["cells"][c]["psi"].items() if v > PSI_ALERT]
    # in-distribution control: each training cell replayed against the others
    ctrl = {}
    for c, g in ref.groupby("cell"):
        other = ref[ref.cell != c]
        ctrl[c] = {f: round(psi(other[f], g.iloc[:k][f]), 2) for f in FEATURES}
    report["in_distribution_psi"] = ctrl
    # Gate control: each training cell scored by a model fit on the other three
    # (champion hyperparameters). The envelope gates should let these through.
    from mlops.core import CellEOLForecaster
    gate_ctrl = {}
    for c in sorted(ref.cell.unique()):
        full = pd.read_csv(os.path.join(ROOT, "data", "capacity.csv"))
        m = CellEOLForecaster.fit(full[full.cell != c], model.hp)
        p = m.predict(full[full.cell == c].sort_values("cycle").iloc[:k])
        gate_ctrl[c] = {"decision": p["decision"], "reasons": p["reasons"]}
    report["in_distribution_gate_control"] = gate_ctrl
    report["psi_note"] = ("Per-cycle PSI is logged as a diagnostic only. With four training cells, "
                          "between-cell spread dominates and in-distribution held-out cells also exceed "
                          "the 0.25 alert, so routing uses the envelope gates in mlops/core.py instead.")
    report["shift_routed_or_rejected"] = sum(v["decision"] in ("route_to_engineer", "reject")
                                             for v in report["cells"].values())
    report["n_shift_cells"] = len(report["cells"])
    out = os.path.join(ROOT, "results", "drift_report.json")
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(report, indent=1))
    with mlflow.start_run(run_name="drift-report"):
        mlflow.log_artifact(out)
        mlflow.log_metric("shift_routed_or_rejected", report["shift_routed_or_rejected"])
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-uri", default="models:/cell-eol-forecaster@champion")
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlruns.db"))
    a = ap.parse_args()
    main(a.model_uri, a.tracking_uri)
