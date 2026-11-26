import json

from src.config import (
    MODELS,
    CONFIGURATIONS,
    get_output_dir,
    get_experiment_params,
    get_numeric_metrics,
)
from src.tracking import log_experiment


def import_experiments():
    imported = 0

    for variant in CONFIGURATIONS:
        output_dir = get_output_dir(variant)
        metrics_path = output_dir / "metrics.json"

        if not metrics_path.exists():
            print(f"Skipping {variant}: metrics not found")
            continue

        with open(metrics_path) as file:
            results = json.load(file)

        params = get_experiment_params(
            variant,
            results,
            source="historical_benchmark",
        )

        metrics = get_numeric_metrics(results)

        print(f"\nImporting {variant}...")

        log_experiment(
            variant=variant,
            params=params,
            metrics=metrics,
            model_path=MODELS[variant],
            output_dir=output_dir,
        )

        imported += 1

    print(f"\nSuccessfully imported {imported} experiments.")


if __name__ == "__main__":
    import_experiments()
