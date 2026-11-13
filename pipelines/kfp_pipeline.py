"""Kubeflow Pipelines (KFP v2) definition of the training workflow.

    data_check -> tune_evaluate_register -> shift_check -> release_note

release_note always runs and states whether the new version was promoted or held
back. (A dsl.If on the `promoted` output is skipped by the KFP local runner, which
cannot resolve step outputs inside conditions.)

Compile to IR YAML for a Kubeflow cluster:
    python pipelines/kfp_pipeline.py --compile pipelines/cell_eol_pipeline.yaml
Run every step locally with the KFP local runner (no cluster needed):
    python pipelines/kfp_pipeline.py --local --trials 60

On a cluster, each component runs in IMAGE (built from the repo Dockerfile)
and MLFLOW_TRACKING_URI must point at a shared MLflow server.
"""

import argparse, os
from typing import NamedTuple

from kfp import compiler, dsl

IMAGE = os.environ.get("PIPELINE_IMAGE", "ghcr.io/zaifmohammed-eng/cell-eol-forecaster:1.0.0")


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def data_check(data_path: str) -> str:
    """Schema and sanity checks; fails the run on bad data. Returns the data SHA-256 prefix."""
    import hashlib
    import numpy as np
    import pandas as pd
    df = pd.read_csv(data_path)
    need = {"cell", "cycle", "capacity_Ah", "rest_before_h", "T_max_C"}
    assert need.issubset(df.columns), f"missing columns {need - set(df.columns)}"
    for c, g in df.groupby("cell"):
        assert np.all(np.diff(g.cycle.values) > 0), f"{c}: cycle index not increasing"
        assert g.capacity_Ah.between(0.05, 5).all(), f"{c}: capacity outside 0.05-5 Ah"
        assert len(g) >= 60, f"{c}: fewer than 60 cycles"
    sha = hashlib.sha256(open(data_path, "rb").read()).hexdigest()[:12]
    print(f"data ok: {df.cell.nunique()} cells, {len(df)} cycles, sha {sha}")
    return sha


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def tune_evaluate_register(trials: int, tracking_uri: str, data_sha: str) -> NamedTuple(
        "Out", [("version", int), ("promoted", bool), ("nested_interval_score", float), ("nested_mae", float)]):
    from collections import namedtuple
    from mlops.train import main
    s = main(trials, tracking_uri, None)
    assert s["registry"]["data_sha256"] == data_sha, "training data changed between steps"
    Out = namedtuple("Out", ["version", "promoted", "nested_interval_score", "nested_mae"])
    return Out(int(s["registry"]["version"]), bool(s["registry"]["promoted"]),
               float(s["nested"]["tuned"]["interval_score"]), float(s["nested"]["tuned"]["mae"]))


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def shift_check(tracking_uri: str, version: int) -> float:
    """Replays unseen-protocol cells; fails if any is forecast instead of routed or rejected."""
    from mlops.monitor import main
    r = main(f"models:/cell-eol-forecaster/{version}", tracking_uri)
    leaked = [c for c, v in r["cells"].items() if v["decision"] == "forecast"]
    ctrl_blocked = [c for c, v in r["in_distribution_gate_control"].items() if v["decision"] != "forecast"]
    assert not leaked, f"shifted cells forecast without review: {leaked}"
    print("in-distribution cells blocked by gates:", ctrl_blocked)
    return r["shift_routed_or_rejected"] / r["n_shift_cells"]


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def release_note(version: int, promoted: bool, nested_interval_score: float, nested_mae: float,
                 shift_route_rate: float) -> str:
    status = "promoted to champion" if promoted else "registered, held back (not better than champion)"
    msg = (f"cell-eol-forecaster v{version} {status}: nested LOCO interval score "
           f"{nested_interval_score:.1f}, EOL MAE {nested_mae:.1f} cycles, shift route rate {shift_route_rate:.0%}")
    print(msg)
    return msg


@dsl.pipeline(name="cell-eol-forecaster-training",
              description="Validate data, tune with nested LOCO, register in MLflow, gate on distribution shift.")
def training_pipeline(data_path: str, tracking_uri: str, trials: int = 60):
    d = data_check(data_path=data_path)
    t = tune_evaluate_register(trials=trials, tracking_uri=tracking_uri, data_sha=d.output)
    s = shift_check(tracking_uri=tracking_uri, version=t.outputs["version"])
    release_note(version=t.outputs["version"], promoted=t.outputs["promoted"],
                 nested_interval_score=t.outputs["nested_interval_score"],
                 nested_mae=t.outputs["nested_mae"], shift_route_rate=s.output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", default="")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--trials", type=int, default=60)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if a.compile:
        compiler.Compiler().compile(training_pipeline, a.compile)
        print("compiled", a.compile)
    if a.local:
        from kfp import local
        local.init(runner=local.SubprocessRunner(use_venv=False), pipeline_root=os.path.join(root, "kfp_local_outputs"))
        training_pipeline(data_path=os.path.join(root, "data", "capacity.csv"),
                          tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{root}/mlruns.db"),
                          trials=a.trials)
