import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.tracking import TRACKING_URI, EXPERIMENT_NAME


REGISTERED_MODEL_NAME = "ssd-mobilenet-v2-edge"
DEPLOYMENT_VARIANT = "official_int8_fast"


def register_model():
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError("MLflow experiment not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"params.variant = '{DEPLOYMENT_VARIANT}' "
            "AND attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No deployment model run found")

    run = runs[0]
    run_id = run.info.run_id

    print(f"Selected run: {run_id}")

    artifacts = client.list_artifacts(run_id, "model")

    model_artifact = next(
        (
            artifact
            for artifact in artifacts
            if artifact.path.endswith(".tflite")
            and not artifact.is_dir
        ),
        None,
    )

    if model_artifact is None:
        raise RuntimeError("TFLite model artifact not found")

    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)

    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise

        client.create_registered_model(
            name=REGISTERED_MODEL_NAME,
            description=(
                "Optimized SSD MobileNet V2 for "
                "resource-constrained edge devices."
            ),
        )

    source = (
        f"{run.info.artifact_uri.rstrip('/')}/"
        f"{model_artifact.path}"
    )

    existing_versions = client.search_model_versions(
        f"name = '{REGISTERED_MODEL_NAME}'"
    )

    version = next(
        (
            item
            for item in existing_versions
            if item.run_id == run_id
        ),
        None,
    )

    if version is None:
        version = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=source,
            run_id=run_id,
            description="INT8 quantization with Fast NMS",
        )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version.version,
        "format",
        "tflite",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version.version,
        "precision",
        "int8",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version.version,
        "nms",
        "fast",
    )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        version.version,
    )

    print("\nModel registered successfully")
    print(f"Name: {REGISTERED_MODEL_NAME}")
    print(f"Version: {version.version}")
    print("Alias: champion")
    print(f"Run ID: {run_id}")
    print(f"Source: {source}")


if __name__ == "__main__":
    register_model()
