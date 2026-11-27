import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.inference import MODEL_PATH
from src.quantize import representative_dataset


OUTPUT_DIR = Path("models/optimized")
DATA_PATH = Path("data/raw/val2017")
ANNOTATIONS_PATH = Path(
    "data/raw/annotations/instances_val2017.json"
)


def representative_dataset_500():
    with open(ANNOTATIONS_PATH) as file:
        annotations = json.load(file)

    images = {
        image["id"]: image["file_name"]
        for image in annotations["images"]
    }

    image_ids = sorted(images)

    rng = random.Random(42)
    rng.shuffle(image_ids)

    # Preserve the original 100 calibration images.
    # Skip the 300 evaluation images.
    calibration_ids = (
        image_ids[:100] + image_ids[400:800]
    )

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


def fixed_shape_converter():
    model = tf.saved_model.load(str(MODEL_PATH))
    inference = model.signatures["serving_default"]

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[1, 320, 320, 3],
                dtype=tf.uint8,
                name="input_tensor"
            )
        ]
    )
    def fixed_inference(input_tensor):
        return inference(input_tensor=input_tensor)

    return tf.lite.TFLiteConverter.from_concrete_functions(
        [fixed_inference.get_concrete_function()],
        model
    )


def quantize(method):
    if method == "fixed":
        converter = fixed_shape_converter()
    else:
        converter = (
            tf.lite.TFLiteConverter.from_saved_model(
                str(MODEL_PATH)
            )
        )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    if method == "dynamic":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]

    elif method in ["fixed", "calib500"]:
        converter.representative_dataset = (
            representative_dataset_500
            if method == "calib500"
            else representative_dataset
        )

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]

        converter.inference_input_type = tf.uint8

    print(f"Applying INT8 {method} quantization...")

    model = converter.convert()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / f"ssd_mobilenet_v2_int8_{method}.tflite"
    )

    output_path.write_bytes(model)

    size = output_path.stat().st_size / (1024 ** 2)

    print(f"Model saved to: {output_path}")
    print(f"Model size: {size:.2f} MiB")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["dynamic", "fixed", "calib500"],
        default="calib500"
    )

    args = parser.parse_args()

    quantize(args.method)


if __name__ == "__main__":
    main()
