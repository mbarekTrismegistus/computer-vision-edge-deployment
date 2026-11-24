import argparse
import json
import time
from pathlib import Path

import cv2
import tensorflow as tf
import numpy as np


MODEL_PATH = Path(
    "models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model"
)

ANNOTATIONS_PATH = Path(
    "data/raw/annotations/instances_val2017.json"
)


def load_model():
    model = tf.saved_model.load(str(MODEL_PATH))
    return model.signatures["serving_default"]


def load_labels():
    with open(ANNOTATIONS_PATH) as file:
        annotations = json.load(file)

    return {
        category["id"]: category["name"]
        for category in annotations["categories"]
    }


def detect(model, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    tensor = tf.expand_dims(tensor, axis=0)

    start = time.perf_counter()
    predictions = model(tensor)

    results = {
        key: value.numpy()
        for key, value in predictions.items()
    }

    latency = (time.perf_counter() - start) * 1000

    return results, latency


def draw_detections(image, results, labels, threshold=0.5):
    height, width = image.shape[:2]

    boxes = results["detection_boxes"][0]
    scores = results["detection_scores"][0]
    classes = results["detection_classes"][0].astype(np.int32)

    count = 0

    for box, score, class_id in zip(boxes, scores, classes):
        if score < threshold:
            continue

        ymin, xmin, ymax, xmax = box

        x1 = int(xmin * width)
        y1 = int(ymin * height)
        x2 = int(xmax * width)
        y2 = int(ymax * height)

        label = labels.get(class_id, "unknown")
        text = f"{label}: {score:.2f}"

        cv2.rectangle(
            image, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            image,
            text,
            (x1, max(15, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        count += 1

    return image, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/detection.jpg")
    args = parser.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        raise ValueError(f"Cannot read image: {args.image}")

    print("Loading model...")
    model = load_model()
    labels = load_labels()

    # Warm-up inference
    detect(model, image)

    # Measured inference
    results, latency = detect(model, image)

    output, count = draw_detections(
        image, results, labels, args.threshold
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, output)

    print(f"Objects detected: {count}")
    print(f"Inference latency: {latency:.2f} ms")
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
