"""Built-in test cases for the public demo.

Every case is real NASA PCoE data or a single, labelled edit of it. Training-cell
cases are scored by a model refit WITHOUT that cell (the same leave-one-cell-out
setup as the evaluation), so the demo never shows an in-sample forecast as if it
were a prediction. Shift and invalid-input cases are scored by the served model.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from mlops.core import CellEOLForecaster
from src.lifetime import true_eol

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["cycle", "capacity_Ah", "rest_before_h", "T_max_C"]

PROTOCOL = {
    "B0005": "24 °C, 2 A discharge (training group)", "B0006": "24 °C, 2 A discharge (training group)",
    "B0007": "24 °C, 2 A discharge (training group)", "B0018": "24 °C, 2 A discharge (training group)",
    "B0029": "43 °C ambient", "B0033": "24 °C, 4 A discharge", "B0036": "24 °C, 4 A discharge",
    "B0045": "4 °C ambient", "B0053": "4 °C ambient",
}


@lru_cache(maxsize=1)
def data():
    tr = pd.read_csv(os.path.join(ROOT, "data", "capacity.csv"))
    sh = pd.read_csv(os.path.join(ROOT, "data", "shift_capacity.csv"))
    return tr, sh


def _cell(df, c):
    return df[df.cell == c].sort_values("cycle").reset_index(drop=True)


def catalog() -> list[dict]:
    tr, sh = data()
    cases = []
    for c in ("B0005", "B0006", "B0007", "B0018"):
        for k in (30, 40, 50):
            cases.append(dict(id=f"holdout-{c}-k{k}", group="Held-out forecast",
                              title=f"{c}, first {k} cycles",
                              detail=f"Model refit on the other three cells; {c} is never seen. {PROTOCOL[c]}.",
                              source=("train", c), k=k, edit=None, expect=["forecast"]))
    cases.append(dict(id="abstain-B0007-k15", group="Abstain & fallback", title="B0007, only 15 cycles",
                      detail="Too little history to forecast: routed to an engineer, interval still shown.",
                      source=("train", "B0007"), k=15, edit=None, expect=["route_to_engineer"]))
    cases.append(dict(id="abstain-jump-B0006", group="Abstain & fallback", title="B0006 with a 25 % capacity drop",
                      detail="Synthetic edit: cycle 20 capacity cut by 25 % (sensor glitch or damage). Routed.",
                      source=("train", "B0006"), k=40, edit="jump", expect=["route_to_engineer"]))
    cases.append(dict(id="reached-B0006", group="Abstain & fallback", title="B0006, 130 cycles (already past EOL)",
                      detail="History already crosses 80 % capacity: the observed EOL is returned, no forecast.",
                      source=("train", "B0006"), k=130, edit=None, expect=["eol_already_reached"]))
    for c in ("B0029", "B0045", "B0053", "B0033", "B0036"):
        cases.append(dict(id=f"shift-{c}", group="Distribution shift (unseen protocol)",
                          title=f"{c}: {PROTOCOL[c]}",
                          detail="Protocol the model never trained on. Must be routed or rejected, never forecast.",
                          source=("shift", c), k=40, edit=None, expect=["route_to_engineer", "reject"]))
    cases.append(dict(id="invalid-zero", group="Invalid input", title="Zero-capacity record",
                      detail="Synthetic edit of B0005: cycle 10 logged as 0 Ah. Rejected.",
                      source=("train", "B0005"), k=40, edit="zero", expect=["reject"]))
    cases.append(dict(id="invalid-order", group="Invalid input", title="Duplicated cycle index",
                      detail="Synthetic edit of B0005: cycle 12 repeated. Rejected.",
                      source=("train", "B0005"), k=40, edit="dup", expect=["reject"]))
    return cases


def _payload(case):
    tr, sh = data()
    df = tr if case["source"][0] == "train" else sh
    full = _cell(df, case["source"][1])
    h = full.iloc[:case["k"]][COLS].copy()
    if case["edit"] == "jump":
        h.loc[19:, "capacity_Ah"] *= 0.75
    elif case["edit"] == "zero":
        h.loc[9, "capacity_Ah"] = 0.0
    elif case["edit"] == "dup":
        h.loc[12, "cycle"] = h.loc[11, "cycle"]
    return full, h


@lru_cache(maxsize=8)
def _holdout_model(cell: str, hp_json: str):
    tr, _ = data()
    base = CellEOLForecaster.from_json(hp_json)
    return CellEOLForecaster.fit(tr[tr.cell != cell], base.hp)


def run(case_id: str, served: CellEOLForecaster) -> dict:
    case = next((c for c in catalog() if c["id"] == case_id), None)
    if case is None:
        raise KeyError(case_id)
    full, h = _payload(case)
    if case["source"][0] == "train" and case["group"] == "Held-out forecast":
        model, scored_by = _holdout_model(case["source"][1], served.to_json()), \
            f"refit without {case['source'][1]} (cells {', '.join(c for c in served.train_cells if c != case['source'][1])})"
    else:
        model, scored_by = served, "served champion"
    p = model.predict(h)
    q = full.capacity_Ah.values
    q0 = float(np.mean(q[:3])) if len(q) >= 3 else float(q[0])
    te = true_eol(q, q0) if case["source"][0] == "train" else float("nan")
    te = None if not np.isfinite(te) else float(te)
    ok = p["decision"] in case["expect"]
    return {
        "case": {k: v for k, v in case.items() if k not in ("source", "edit")},
        "scored_by": scored_by, "prediction": p, "true_eol": te,
        "abs_error": None if (te is None or p["eol_p50"] is None or p["decision"] != "forecast")
        else abs(p["eol_p50"] - te),
        "covered": None if (te is None or p["eol_p05"] is None or p["decision"] != "forecast")
        else bool(p["eol_p05"] <= te <= p["eol_p95"]),
        "pass": ok,
        "series": {"cycle": full.cycle.tolist(), "capacity_Ah": full.capacity_Ah.round(5).tolist(),
                   "observed_cycle": h.cycle.tolist(), "observed_capacity_Ah": h.capacity_Ah.round(5).tolist(),
                   "q0_Ah": q0},
        "request": {"cell_id": case["source"][1],
                    "cycles": h.astype(object).where(h.notna(), None).to_dict("records")},
    }
