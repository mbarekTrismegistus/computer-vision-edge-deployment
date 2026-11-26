import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(
    "models/optimized/ssd_mobilenet_v2_official_fp32.tflite"
)


def load_official_model(model_path=MODEL_PATH):
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    return {
        "runner": interpreter.get_signature_runner("serving_default"),
        "input_details": interpreter.get_input_details()[0],
    }


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))
    image = image.astype(np.float32)
    image = image * (2.0 / 255.0) - 1.0

    return np.expand_dims(image, axis=0)


def detect_official(model, image):
    tensor = preprocess(image)

    input_details = model["input_details"]
    dtype = input_details["dtype"]

    if np.issubdtype(dtype, np.integer):
        scale, zero_point = input_details["quantization"]

        limits = np.iinfo(dtype)

        tensor = np.clip(
            np.round(tensor / scale + zero_point),
            limits.min,
            limits.max,
        ).astype(dtype)

    else:
        tensor = tensor.astype(dtype)

    start = time.perf_counter()

    outputs = model["runner"](input=tensor)

    latency = (time.perf_counter() - start) * 1000

    results = {
        "detection_boxes": outputs["output_3"],
        "detection_classes": outputs["output_2"] + 1,
        "detection_scores": outputs["output_1"],
        "num_detections": outputs["output_0"],
    }

    return results, latency
