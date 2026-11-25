import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.inference import MODEL_PATH


OUTPUT_DIR = Path("models/optimized")
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

    calibration_ids = image_ids[:CALIBRATION_SIZE]

    for image_id in calibration_ids:
        image_path = DATA_PATH / images[image_id]

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Cannot read {image_path}"
            )

        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(image, (320, 320))

        tensor = np.expand_dims(
            image, axis=0
        ).astype(np.uint8)

        yield {"input_tensor": tensor}


def quantize(precision):
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(MODEL_PATH)
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    if precision == "fp16":
        converter.target_spec.supported_types = [
            tf.float16
        ]

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]

    elif precision == "int8":
        converter.representative_dataset = (
            representative_dataset
        )

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]

        converter.inference_input_type = tf.uint8

    print(f"Quantizing model to {precision.upper()}...")

    model = converter.convert()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / f"ssd_mobilenet_v2_{precision}.tflite"
    )

    output_path.write_bytes(model)

    size = output_path.stat().st_size / (1024 ** 2)

    print(f"Model saved to: {output_path}")
    print(f"Model size: {size:.2f} MiB")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize SSD MobileNet V2"
    )

    parser.add_argument(
        "--precision",
        choices=["fp16", "int8"],
        default="fp16"
    )

    args = parser.parse_args()

    quantize(args.precision)


if __name__ == "__main__":
    main()
