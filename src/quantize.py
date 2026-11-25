from pathlib import Path

import tensorflow as tf

from src.inference import MODEL_PATH


OUTPUT_DIR = Path("models/optimized")


def quantize_fp16():
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(MODEL_PATH)
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.target_spec.supported_types = [
        tf.float16
    ]

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    print("Quantizing model to FP16...")

    model = converter.convert()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "ssd_mobilenet_v2_fp16.tflite"

    output_path.write_bytes(model)

    size = output_path.stat().st_size / (1024 ** 2)

    print(f"Model saved to: {output_path}")
    print(f"Model size: {size:.2f} MiB")


if __name__ == "__main__":
    quantize_fp16()
