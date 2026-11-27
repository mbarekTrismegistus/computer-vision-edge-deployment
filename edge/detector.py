import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

from edge.labels import COCO_LABELS


class Detector:
    def __init__(self, model_path: str | Path, threads: int = 2):
        self.model_path = Path(model_path)
        self.threads = threads
        self.lock = Lock()

        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)

        if threads < 1:
            raise ValueError("Thread count must be positive")

        cv2.setNumThreads(1)

        self.interpreter = Interpreter(
            model_path=str(self.model_path),
            num_threads=threads,
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()[0]
        )

        self.runner = self.interpreter.get_signature_runner(
            "serving_default"
        )

        _, self.height, self.width, _ = map(
            int, self.input_details["shape"]
        )

        self.warmup()

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.width, self.height),
        )

        image = image.astype(np.float32)
        image = image * (2.0 / 255.0) - 1.0

        dtype = self.input_details["dtype"]

        if np.issubdtype(dtype, np.integer):
            scale, zero_point = (
                self.input_details["quantization"]
            )

            if scale <= 0:
                raise ValueError("Invalid quantization scale")

            limits = np.iinfo(dtype)

            image = np.clip(
                np.round(image / scale + zero_point),
                limits.min,
                limits.max,
            ).astype(dtype)

        else:
            image = image.astype(dtype)

        return np.expand_dims(image, axis=0)

    def warmup(self, runs=5):
        image = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8,
        )

        tensor = self.preprocess(image)

        with self.lock:
            for _ in range(runs):
                self.runner(input=tensor)

    def predict(self, image, threshold=0.5):
        if image is None or image.size == 0:
            raise ValueError("Invalid image")

        if not 0 <= threshold <= 1:
            raise ValueError("Invalid confidence threshold")

        processing_start = time.perf_counter()

        height, width = image.shape[:2]

        tensor = self.preprocess(image)

        with self.lock:
            inference_start = time.perf_counter()

            outputs = self.runner(input=tensor)

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000

        boxes = outputs["output_3"][0]
        classes = outputs["output_2"][0] + 1
        scores = outputs["output_1"][0]

        count = min(
            int(outputs["output_0"].reshape(-1)[0]),
            len(boxes),
            len(classes),
            len(scores),
        )

        detections = []

        for index in range(count):
            confidence = float(scores[index])

            if confidence < threshold:
                continue

            class_id = int(classes[index])

            if class_id not in COCO_LABELS:
                continue

            ymin, xmin, ymax, xmax = np.clip(
                boxes[index], 0, 1
            )

            detections.append({
                "class_id": class_id,
                "label": COCO_LABELS[class_id],
                "confidence": round(confidence, 4),
                "bbox": {
                    "xmin": int(xmin * width),
                    "ymin": int(ymin * height),
                    "xmax": int(xmax * width),
                    "ymax": int(ymax * height),
                },
            })

        processing_ms = (
            time.perf_counter() - processing_start
        ) * 1000

        return {
            "model": self.model_path.stem,
            "inference_ms": round(inference_ms, 2),
            "processing_ms": round(processing_ms, 2),
            "image_width": width,
            "image_height": height,
            "count": len(detections),
            "detections": detections,
        }

    def info(self):
        return {
            "name": self.model_path.stem,
            "runtime": "tflite-runtime",
            "threads": self.threads,
            "size_mb": round(
                self.model_path.stat().st_size / 1024**2,
                2,
            ),
            "input_shape": (
                self.input_details["shape"].tolist()
            ),
            "input_dtype": np.dtype(
                self.input_details["dtype"]
            ).name,
        }
