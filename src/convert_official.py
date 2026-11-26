from pathlib import Path

import tensorflow as tf

MODEL_PATH = Path("models/tflite_export/saved_model")
OUTPUT_PATH = Path(
    "models/optimized/ssd_mobilenet_v2_official_fp32.tflite"
)


def convert():
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(MODEL_PATH)
    )

    converter.optimizations = []
    converter.allow_custom_ops = True

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    model = converter.convert()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(model)

    print(f"Model saved to: {OUTPUT_PATH}")
    print(f"Model size: {OUTPUT_PATH.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    convert()
