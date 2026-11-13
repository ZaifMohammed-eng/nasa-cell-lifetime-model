"""Export a registered model to a single JSON file for serving without MLflow.

    python -m mlops.export --model-uri models:/cell-eol-forecaster@champion --out serve/model/champion.json

Writes champion.json (the forecaster) and champion.meta.json (registry version,
run id, nested evaluation metrics) so a public demo can show what it is serving.
"""
import argparse, json, os

import mlflow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model-uri", default="models:/cell-eol-forecaster@champion")
ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlruns.db"))
ap.add_argument("--out", default=os.path.join(ROOT, "serve", "model", "champion.json"))
a = ap.parse_args()

mlflow.set_tracking_uri(a.tracking_uri)
client = mlflow.MlflowClient()
m = mlflow.pyfunc.load_model(a.model_uri).unwrap_python_model().m
name, alias = a.model_uri[len("models:/"):].split("@")
mv = client.get_model_version_by_alias(name, alias)
run = client.get_run(mv.run_id)
meta = {"model": name, "alias": alias, "version": mv.version, "run_id": mv.run_id,
        "git_sha": run.data.tags.get("git_sha"), "data_sha256": run.data.tags.get("data_sha256"),
        "nested_loco": {k.replace("nested_", ""): v for k, v in run.data.metrics.items() if k.startswith("nested_")},
        "baseline_fleet_mean_mae": run.data.metrics.get("baseline_fleet_mean_mae")}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write(m.to_json())
json.dump(meta, open(a.out.replace(".json", ".meta.json"), "w"), indent=1)
print(f"exported {name} v{mv.version} -> {a.out}")
