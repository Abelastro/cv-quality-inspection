from pydantic import BaseModel
from typing import Optional
from enum import Enum


class DefectType(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    DISCOLORATION = "discoloration"
    NONE = "none"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class DetectionResult(BaseModel):
    image_path: str
    detections: list[BoundingBox]
    processing_time_ms: float


class Measurement(BaseModel):
    width_px: float
    height_px: float
    area_px: float
    aspect_ratio: float


class QualityResult(BaseModel):
    image_path: str
    detections: list[BoundingBox]
    measurements: list[Measurement]
    quality_score: float
    defects_found: int
    passed: bool
