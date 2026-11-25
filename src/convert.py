from pathlib import Path
import tensorflow as tf

from src.inference import MODEL_PATH


OUTPUT_PATH = Path("models/optimized/ssd_mobilenet_v2_fp32.tflite")


def convert():
    print("Loading TensorFlow SavedModel...")

    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(MODEL_PATH)
    )

    # Convert without quantization.
    converter.optimizations = []

    # Only allow native TensorFlow Lite operations.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    print("Converting to TensorFlow Lite...")

    tflite_model = converter.convert()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_bytes(tflite_model)

    size = OUTPUT_PATH.stat().st_size / (1024 ** 2)

    print(f"Model saved to: {OUTPUT_PATH}")
    print(f"Model size: {size:.2f} MiB")


if __name__ == "__main__":
    convert()
