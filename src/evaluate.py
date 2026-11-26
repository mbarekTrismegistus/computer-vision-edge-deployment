import argparse
import json
import random
import resource
from pathlib import Path

import cv2
import numpy as np
import psutil
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.inference import load_model, detect
from src.tflite_inference import (
    load_tflite_model,
    detect_tflite,
)
from src.official_inference import (
    load_official_model,
    detect_official,
)
from src.config import (
    MODELS,
    DATA_PATH,
    ANNOTATIONS_PATH,
    CALIBRATION_SEED,
    WARMUP_RUNS,
    get_output_dir,
    get_experiment_params,
    get_numeric_metrics,
)


def model_size_mb(path):
    path = Path(path)

    if path.is_file():
        return path.stat().st_size / (1024 ** 2)

    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    ) / (1024 ** 2)


def select_evaluation_images(image_ids, sample_size, seed):
    calibration_ids = sorted(image_ids)

    calibration_rng = random.Random(CALIBRATION_SEED)
    calibration_rng.shuffle(calibration_ids)

    # Exclude the calibration images used by all model variants.
    reserved_ids = set(
        calibration_ids[:100] + calibration_ids[400:800]
    )

    evaluation_ids = sorted(image_ids)

    evaluation_rng = random.Random(seed)
    evaluation_rng.shuffle(evaluation_ids)

    evaluation_ids = [
        image_id
        for image_id in evaluation_ids
        if image_id not in reserved_ids
    ]

    if not 1 <= sample_size <= len(evaluation_ids):
        raise ValueError(
            f"Sample size must be between 1 and "
            f"{len(evaluation_ids)}"
        )

    return evaluation_ids[:sample_size]


def evaluate(sample_size, seed, variant="savedmodel"):
    coco = COCO(str(ANNOTATIONS_PATH))
    category_ids = set(coco.getCatIds())

    image_ids = select_evaluation_images(
        coco.getImgIds(),
        sample_size,
        seed,
    )

    model_path = MODELS[variant]

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if variant == "savedmodel":
        model = load_model()
        inference = detect
        backend = "savedmodel"

    elif variant.startswith("official_"):
        model = load_official_model(model_path)
        inference = detect_official
        backend = "tflite"

    else:
        model = load_tflite_model(model_path)
        inference = detect_tflite
        backend = "tflite"

    process = psutil.Process()

    detections = []
    latencies = []

    # Warm-up inference
    first_image = coco.loadImgs(image_ids[0])[0]

    image = cv2.imread(
        str(DATA_PATH / first_image["file_name"])
    )

    if image is None:
        raise ValueError("Cannot read warm-up image")

    for _ in range(WARMUP_RUNS):
        inference(model, image)

    # Benchmark
    for image_id in tqdm(
        image_ids,
        desc=f"Evaluating {variant}"
    ):
        metadata = coco.loadImgs(image_id)[0]

        image = cv2.imread(
            str(DATA_PATH / metadata["file_name"])
        )

        if image is None:
            raise ValueError(
                f"Cannot read {metadata['file_name']}"
            )

        height, width = image.shape[:2]

        results, latency = inference(model, image)
        latencies.append(latency)

        boxes = results["detection_boxes"][0]
        scores = results["detection_scores"][0]
        classes = results["detection_classes"][0]

        count = int(results["num_detections"][0])

        for box, score, class_id in zip(
            boxes[:count],
            scores[:count],
            classes[:count]
        ):
            class_id = int(class_id)

            if class_id not in category_ids:
                continue

            ymin, xmin, ymax, xmax = box

            detections.append({
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [
                    float(xmin * width),
                    float(ymin * height),
                    float((xmax - xmin) * width),
                    float((ymax - ymin) * height),
                ],
                "score": float(score),
            })

    if not detections:
        raise RuntimeError("No detections produced")

    predictions = coco.loadRes(detections)

    evaluator = COCOeval(coco, predictions, "bbox")
    evaluator.params.imgIds = image_ids

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    mean_latency = float(np.mean(latencies))

    metrics = {
        "model": f"ssd_mobilenet_v2_{variant}",
        "backend": backend,
        "precision": (
            "fp32" if variant == "savedmodel"
            else variant
        ),
        "sample_size": len(image_ids),
        "seed": seed,
        "map_50_95": float(evaluator.stats[0]),
        "map_50": float(evaluator.stats[1]),
        "map_75": float(evaluator.stats[2]),
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": float(
            np.percentile(latencies, 95)
        ),
        "inference_fps": 1000 / mean_latency,
        "rss_mb": process.memory_info().rss / (1024 ** 2),
        "peak_rss_mb": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss / 1024,
        "model_size_mb": model_size_mb(model_path),
    }

    return metrics, image_ids, detections


def save_results(output, metrics, image_ids, detections):
    output.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "metrics.json": metrics,
        "image_ids.json": image_ids,
        "predictions.json": detections,
    }

    for filename, content in artifacts.items():
        with open(output / filename, "w") as file:
            json.dump(content, file, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SSD MobileNet V2 on COCO"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=300
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="savedmodel"
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking"
    )

    args = parser.parse_args()

    if not 1 <= args.samples <= 4500:
        parser.error(
            "--samples must be between 1 and 4500"
        )

    metrics, image_ids, detections = evaluate(
        args.samples,
        args.seed,
        args.model
    )

    output = get_output_dir(args.model)

    save_results(
        output,
        metrics,
        image_ids,
        detections,
    )

    print(f"\n{args.model} evaluation metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved to {output}")

    if not args.no_mlflow:
        from src.tracking import log_experiment

        params = get_experiment_params(
            args.model,
            metrics,
            source="evaluation",
        )

        params["evaluation_split"] = "exclude_all_calibration"

        numeric_metrics = get_numeric_metrics(metrics)

        print("\nLogging evaluation to MLflow...")

        log_experiment(
            variant=args.model,
            params=params,
            metrics=numeric_metrics,
            model_path=MODELS[args.model],
            output_dir=output,
        )


if __name__ == "__main__":
    main()
