import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.official_inference import preprocess

MODEL_PATH = Path("models/tflite_export/saved_model")

OUTPUT_PATH = Path(
    "models/optimized/ssd_mobilenet_v2_official_int8.tflite"
)

DATA_PATH = Path("data/raw/val2017")

ANNOTATIONS_PATH = Path(
    "data/raw/annotations/instances_val2017.json"
)

CALIBRATION_SIZE = 100
SEED = 42


def representative_dataset():
    with open(ANNOTATIONS_PATH) as file:
        annotations = json.load(file)

    images = {
        image["id"]: image["file_name"]
        for image in annotations["images"]
    }

    image_ids = sorted(images)

    rng = random.Random(SEED)
    rng.shuffle(image_ids)

    for image_id in image_ids[:CALIBRATION_SIZE]:
        image_path = DATA_PATH / images[image_id]
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Cannot read {image_path}")

        yield {"input": preprocess(image)}


def quantize():
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(MODEL_PATH)
    )

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset

    converter.allow_custom_ops = True

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32

    print("Quantizing official SSD model to INT8...")

    model = converter.convert()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(model)

    print(f"Model saved to: {OUTPUT_PATH}")
    print(f"Model size: {OUTPUT_PATH.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    quantize()
