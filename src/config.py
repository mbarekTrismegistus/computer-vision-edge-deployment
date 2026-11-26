from pathlib import Path


DATA_PATH = Path("data/raw/val2017")
ANNOTATIONS_PATH = Path(
    "data/raw/annotations/instances_val2017.json"
)

BASE_MODEL_PATH = Path(
    "models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model"
)

OPTIMIZED_PATH = Path("models/optimized")

CALIBRATION_SEED = 42
WARMUP_RUNS = 5

MODELS = {
    "savedmodel": BASE_MODEL_PATH,
    "fp32": OPTIMIZED_PATH / "ssd_mobilenet_v2_fp32.tflite",
    "fp16": OPTIMIZED_PATH / "ssd_mobilenet_v2_fp16.tflite",
    "int8": OPTIMIZED_PATH / "ssd_mobilenet_v2_int8.tflite",
    "int8_dynamic": OPTIMIZED_PATH / "ssd_mobilenet_v2_int8_dynamic.tflite",
    "int8_fixed": OPTIMIZED_PATH / "ssd_mobilenet_v2_int8_fixed.tflite",
    "int8_calib500": OPTIMIZED_PATH / "ssd_mobilenet_v2_int8_calib500.tflite",
    "official_fp32": OPTIMIZED_PATH / "ssd_mobilenet_v2_official_fp32.tflite",
    "official_int8": OPTIMIZED_PATH / "ssd_mobilenet_v2_official_int8.tflite",
    "official_int8_fast": OPTIMIZED_PATH / "ssd_mobilenet_v2_official_int8_fast.tflite",
}

CONFIGURATIONS = {
    "savedmodel": {
        "precision": "FP32",
        "export_method": "original",
        "quantization": "none",
        "calibration_size": 0,
        "input_shape": "dynamic",
        "nms": "original",
    },
    "fp32": {
        "precision": "FP32",
        "export_method": "direct",
        "quantization": "none",
        "calibration_size": 0,
        "input_shape": "320x320",
        "nms": "original",
    },
    "fp16": {
        "precision": "FP16",
        "export_method": "direct",
        "quantization": "float16",
        "calibration_size": 0,
        "input_shape": "320x320",
        "nms": "original",
    },
    "int8": {
        "precision": "INT8",
        "export_method": "direct",
        "quantization": "post_training",
        "calibration_size": 100,
        "input_shape": "320x320",
        "nms": "original",
    },
    "int8_dynamic": {
        "precision": "mixed",
        "export_method": "direct",
        "quantization": "dynamic_range",
        "calibration_size": 0,
        "input_shape": "320x320",
        "nms": "original",
    },
    "int8_fixed": {
        "precision": "INT8",
        "export_method": "direct_fixed",
        "quantization": "post_training",
        "calibration_size": 100,
        "input_shape": "320x320",
        "nms": "original",
    },
    "int8_calib500": {
        "precision": "INT8",
        "export_method": "direct",
        "quantization": "post_training",
        "calibration_size": 500,
        "input_shape": "320x320",
        "nms": "original",
    },
    "official_fp32": {
        "precision": "FP32",
        "export_method": "official",
        "quantization": "none",
        "calibration_size": 0,
        "input_shape": "300x300",
        "nms": "regular",
    },
    "official_int8": {
        "precision": "INT8",
        "export_method": "official",
        "quantization": "post_training",
        "calibration_size": 100,
        "input_shape": "300x300",
        "nms": "regular",
    },
    "official_int8_fast": {
        "precision": "INT8",
        "export_method": "official",
        "quantization": "post_training",
        "calibration_size": 100,
        "input_shape": "300x300",
        "nms": "fast",
    },
}

METRIC_KEYS = [
    "map_50_95",
    "map_50",
    "map_75",
    "mean_latency_ms",
    "p95_latency_ms",
    "inference_fps",
    "rss_mb",
    "peak_rss_mb",
    "model_size_mb",
]


def get_output_dir(variant):
    return Path(
        "outputs/baseline"
        if variant == "savedmodel"
        else f"outputs/tflite_{variant}"
    )


def get_experiment_params(variant, results, source):
    return {
        **CONFIGURATIONS[variant],
        "variant": variant,
        "backend": results["backend"],
        "sample_size": results["sample_size"],
        "seed": results["seed"],
        "calibration_seed": CALIBRATION_SEED,
        "warmup_runs": WARMUP_RUNS,
        "device": "CPU",
        "source": source,
    }


def get_numeric_metrics(results):
    return {
        key: results[key]
        for key in METRIC_KEYS
        if key in results
    }
