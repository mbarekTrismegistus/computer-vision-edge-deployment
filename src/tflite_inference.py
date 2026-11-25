import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


TFLITE_MODEL_PATH = Path(
    "models/optimized/ssd_mobilenet_v2_fp32.tflite"
)


def load_tflite_model(model_path=TFLITE_MODEL_PATH):
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    input_details = interpreter.get_input_details()[0]

    interpreter.resize_tensor_input(
        input_details["index"],
        [1, 320, 320, 3]
    )

    interpreter.allocate_tensors()

    return interpreter.get_signature_runner(
        "serving_default"
    )


def detect_tflite(model, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 320))
    tensor = np.expand_dims(image, axis=0)

    start = time.perf_counter()

    outputs = model(input_tensor=tensor)

    latency = (time.perf_counter() - start) * 1000

    return outputs, latency
