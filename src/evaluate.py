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

from src.inference import load_model, detect, MODEL_PATH
from src.tflite_inference import (
    load_tflite_model,
    detect_tflite,
    TFLITE_MODEL_PATH,
)


DATA_PATH = Path("data/raw/val2017")
ANNOTATIONS_PATH = Path(
    "data/raw/annotations/instances_val2017.json"
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


def evaluate(sample_size, seed, backend="savedmodel"):
    coco = COCO(str(ANNOTATIONS_PATH))
    image_ids = sorted(coco.getImgIds())
    category_ids = set(coco.getCatIds())

    rng = random.Random(seed)
    rng.shuffle(image_ids)

    # Reserve the first 100 images for calibration.
    image_ids = image_ids[100:100 + sample_size]

    if backend == "tflite":
        model = load_tflite_model()
        inference = detect_tflite
        model_path = TFLITE_MODEL_PATH
    else:
        model = load_model()
        inference = detect
        model_path = MODEL_PATH

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

    for _ in range(5):
        inference(model, image)

    # Evaluate selected images
    for image_id in tqdm(image_ids, desc=f"Evaluating {backend}"):
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

    # COCO accuracy evaluation
    predictions = coco.loadRes(detections)

    evaluator = COCOeval(coco, predictions, "bbox")
    evaluator.params.imgIds = image_ids

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # Performance metrics
    mean_latency = float(np.mean(latencies))

    metrics = {
        "model": f"ssd_mobilenet_v2_{backend}",
        "backend": backend,
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
        "--backend",
        choices=["savedmodel", "tflite"],
        default="savedmodel"
    )

    args = parser.parse_args()

    if not 1 <= args.samples <= 4900:
        parser.error("--samples must be between 1 and 4900")

    metrics, image_ids, detections = evaluate(
        args.samples,
        args.seed,
        args.backend
    )

    output = Path(
        "outputs/baseline"
        if args.backend == "savedmodel"
        else "outputs/tflite_fp32"
    )

    output.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "metrics.json": metrics,
        "image_ids.json": image_ids,
        "predictions.json": detections,
    }

    for filename, content in artifacts.items():
        with open(output / filename, "w") as file:
            json.dump(content, file, indent=2)

    print(f"\n{args.backend} evaluation metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
