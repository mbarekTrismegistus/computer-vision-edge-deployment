## FP32 Benchmark Results

Evaluation performed on 300 COCO 2017 validation images using random seed 42.

| Metric | SavedModel FP32 | TFLite FP32 |
|---|---:|---:|
| mAP@50:95 | 23.05% | 23.39% |
| mAP@50 | 36.74% | 37.08% |
| Mean latency | 79.38 ms | 82.36 ms |
| P95 latency | 124.66 ms | 110.99 ms |
| FPS | 12.60 | 12.14 |
| RAM usage | 1290.17 MiB | 789.05 MiB |
| Model size | 32.19 MiB | 23.72 MiB |

TFLite conversion reduced model size by 26.31% and process memory consumption by 38.84%, while maintaining comparable detection accuracy.

Performance measurements are preliminary and include runtime and evaluation overhead.
