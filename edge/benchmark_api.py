import argparse
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import psutil


IMAGE_PATH = Path("data/raw/val2017/000000000139.jpg")
OUTPUT_PATH = Path("outputs/api")
BOUNDARY = "edge-benchmark-boundary"


def build_body(image_path):
    header = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{image_path.name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()

    footer = f"\r\n--{BOUNDARY}--\r\n".encode()

    return header + image_path.read_bytes() + footer


def send_request(url, body):
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": (
                f"multipart/form-data; boundary={BOUNDARY}"
            )
        },
        method="POST",
    )

    start = time.perf_counter()

    with urlopen(request, timeout=30) as response:
        result = json.loads(response.read())

    latency = (time.perf_counter() - start) * 1000

    return result, latency


def benchmark(url, requests, pid):
    body = build_body(IMAGE_PATH)
    process = psutil.Process(pid)

    print("Running warm-up requests...")

    for _ in range(5):
        send_request(url, body)

    latencies = []
    inference_times = []
    processing_times = []
    memory_samples = []
    failures = 0

    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()

    print(f"Sending {requests} requests...")

    for index in range(requests):
        try:
            result, latency = send_request(url, body)

            latencies.append(latency)
            inference_times.append(result["inference_ms"])
            processing_times.append(result["processing_ms"])

            memory_samples.append(
                process.memory_info().rss / 1024**2
            )

        except Exception as error:
            failures += 1
            print(f"Request {index + 1} failed: {error}")

        if (index + 1) % 10 == 0:
            print(f"Completed {index + 1}/{requests}")

    wall_time = time.perf_counter() - wall_start
    cpu_end = process.cpu_times()

    cpu_time = (
        cpu_end.user - cpu_start.user
        + cpu_end.system - cpu_start.system
    )

    if not latencies:
        raise RuntimeError("All requests failed")

    mean_latency = statistics.mean(latencies)
    mean_processing = statistics.mean(processing_times)

    metrics = {
        "total_requests": requests,
        "successful_requests": len(latencies),
        "failed_requests": failures,
        "success_rate_percent": len(latencies) / requests * 100,
        "mean_http_latency_ms": mean_latency,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "mean_inference_ms": statistics.mean(inference_times),
        "mean_processing_ms": mean_processing,
        "mean_http_overhead_ms": mean_latency - mean_processing,
        "requests_per_second": len(latencies) / wall_time,
        "mean_rss_mb": statistics.mean(memory_samples),
        "max_sampled_rss_mb": max(memory_samples),
        "cpu_percent": cpu_time / wall_time * 100,
        "duration_seconds": wall_time,
    }

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_PATH / "benchmark.json"

    with open(output_file, "w") as file:
        json.dump(metrics, file, indent=2)

    print("\nAPI benchmark results:")
    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the edge inference API"
    )

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/predict?threshold=0.5",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--pid",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be positive")

    benchmark(args.url, args.requests, args.pid)


if __name__ == "__main__":
    main()
