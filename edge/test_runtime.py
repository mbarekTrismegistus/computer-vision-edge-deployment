import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
from tflite_runtime.interpreter import Interpreter


MODEL_PATH = Path(
    "models/optimized/ssd_mobilenet_v2_official_int8_fast.tflite"
)

DATA_PATH = Path("data/raw/val2017")
OUTPUT_PATH = Path("outputs/runtime")

WARMUP_RUNS = 5
BENCHMARK_RUNS = 100


def benchmark(threads):
    cv2.setNumThreads(1)

    process = psutil.Process()

    start = time.perf_counter()

    interpreter = Interpreter(
        model_path=str(MODEL_PATH),
        num_threads=threads,
    )

    interpreter.allocate_tensors()

    load_time = (time.perf_counter() - start) * 1000

    input_details = interpreter.get_input_details()[0]
    runner = interpreter.get_signature_runner("serving_default")

    image_path = next(iter(sorted(DATA_PATH.glob("*.jpg"))))
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))
    image = image.astype(np.float32)
    image = image * (2.0 / 255.0) - 1.0

    scale, zero_point = input_details["quantization"]

    dtype = input_details["dtype"]
    limits = np.iinfo(dtype)

    tensor = np.clip(
        np.round(image / scale + zero_point),
        limits.min,
        limits.max,
    ).astype(dtype)

    tensor = np.expand_dims(tensor, axis=0)

    for _ in range(WARMUP_RUNS):
        runner(input=tensor)

    latencies = []

    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        runner(input=tensor)

        latencies.append(
            (time.perf_counter() - start) * 1000
        )

    wall_time = time.perf_counter() - wall_start
    cpu_end = process.cpu_times()

    cpu_time = (
        cpu_end.user - cpu_start.user
        + cpu_end.system - cpu_start.system
    )

    mean_latency = float(np.mean(latencies))

    metrics = {
        "runtime": "tflite-runtime",
        "threads": threads,
        "benchmark_runs": BENCHMARK_RUNS,
        "warmup_runs": WARMUP_RUNS,
        "image": image_path.name,
        "model_size_mb": MODEL_PATH.stat().st_size / 1024**2,
        "load_time_ms": load_time,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "inference_fps": 1000 / mean_latency,
        "rss_mb": process.memory_info().rss / 1024**2,
        "cpu_percent": cpu_time / wall_time * 100,
    }

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_PATH / f"threads_{threads}.json"

    with open(output_file, "w") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark lightweight TFLite runtime"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")

    benchmark(args.threads)


if __name__ == "__main__":
    main()
