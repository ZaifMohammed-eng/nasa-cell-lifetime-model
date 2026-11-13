"""Submit the compiled pipeline to a Kubeflow Pipelines endpoint.

    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80   # in another terminal
    python pipelines/submit.py --host http://localhost:8080
"""
import argparse

from kfp import Client

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="http://localhost:8080")
ap.add_argument("--package", default="pipelines/cell_eol_pipeline.yaml")
ap.add_argument("--tracking-uri", default="http://mlflow.kubeflow.svc.cluster.local:5000")
ap.add_argument("--trials", type=int, default=60)
a = ap.parse_args()

run = Client(host=a.host).create_run_from_pipeline_package(
    a.package,
    arguments={"data_path": "/app/data/capacity.csv", "tracking_uri": a.tracking_uri, "trials": a.trials},
    run_name="cell-eol-training",
    experiment_name="cell-eol-forecaster",
    enable_caching=False,  # always retrain; cached steps would skip tuning
)
print("submitted run", run.run_id)
