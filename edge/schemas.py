from pydantic import BaseModel


class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox


class PredictionResponse(BaseModel):
    model: str
    inference_ms: float
    processing_ms: float
    image_width: int
    image_height: int
    count: int
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelResponse(BaseModel):
    name: str
    runtime: str
    threads: int
    size_mb: float
    input_shape: list[int]
    input_dtype: str
