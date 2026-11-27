import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query
from fastapi import Request, UploadFile

from edge.detector import Detector
from edge.schemas import (
    HealthResponse,
    ModelResponse,
    PredictionResponse,
)


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/optimized/ssd_mobilenet_v2_official_int8_fast.tflite",
)

NUM_THREADS = int(os.getenv("NUM_THREADS", "2"))

MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector(
        model_path=MODEL_PATH,
        threads=NUM_THREADS,
    )

    yield

    del app.state.detector


app = FastAPI(
    title="Edge Object Detection API",
    description="Lightweight SSD MobileNet V2 inference service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    loaded = hasattr(request.app.state, "detector")

    return {
        "status": "healthy" if loaded else "unavailable",
        "model_loaded": loaded,
    }


@app.get("/model", response_model=ModelResponse)
def model_info(request: Request):
    return request.app.state.detector.info()


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Query(default=0.5, ge=0, le=1),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format",
        )

    content = file.file.read(MAX_IMAGE_SIZE + 1)

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MiB limit",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty image",
        )

    buffer = np.frombuffer(content, dtype=np.uint8)

    try:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error:
        image = None

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image",
        )

    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail="Image dimensions exceed the limit",
        )

    return request.app.state.detector.predict(
        image,
        threshold=threshold,
    )
