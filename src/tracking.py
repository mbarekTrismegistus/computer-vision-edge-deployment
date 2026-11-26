import os
from pathlib import Path

import mlflow


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

EXPERIMENT_NAME = "SSD MobileNet V2 Edge Optimization"


def setup_tracking():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_experiment(variant, params, metrics, model_path, output_dir):
    setup_tracking()

    model_path = Path(model_path)
    output_dir = Path(output_dir)

    with mlflow.start_run(run_name=variant) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.set_tags({
            "architecture": "SSD MobileNet V2",
            "task": "object_detection",
            "dataset": "COCO 2017",
            "target": "edge_deployment",
        })

        for filename in [
            "metrics.json",
            "predictions.json",
            "image_ids.json",
        ]:
            artifact = output_dir / filename

            if artifact.exists():
                mlflow.log_artifact(
                    str(artifact),
                    artifact_path="evaluation"
                )

        if model_path.is_dir():
            mlflow.log_artifacts(
                str(model_path),
                artifact_path="model"
            )
        elif model_path.is_file():
            mlflow.log_artifact(
                str(model_path),
                artifact_path="model"
            )
        else:
            raise FileNotFoundError(model_path)

        print(f"MLflow run ID: {run.info.run_id}")

        return run.info.run_id
