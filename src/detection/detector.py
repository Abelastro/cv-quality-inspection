import cv2
import numpy as np
from typing import Optional
from ..models.schemas import BoundingBox, DetectionResult
import time


class Detector:
    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.5):
        self.model_path = model_path
        self.confidence = confidence
        self.model = None

    def _load_model(self):
        if self.model_path:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

    def detect(self, image_path: str) -> DetectionResult:
        start_time = time.time()

        if self.model is None:
            self._load_model()

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        if self.model:
            results = self.model(img, conf=self.confidence)
            detections = self._parse_results(results)
        else:
            detections = self._mock_detection(img)

        processing_time = (time.time() - start_time) * 1000

        return DetectionResult(
            image_path=image_path,
            detections=detections,
            processing_time_ms=processing_time,
        )

    def _parse_results(self, results) -> list[BoundingBox]:
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append(BoundingBox(
                    x1=float(box.xyxy[0][0]),
                    y1=float(box.xyxy[0][1]),
                    x2=float(box.xyxy[0][2]),
                    y2=float(box.xyxy[0][3]),
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                    class_name=result.names[int(box.cls[0])],
                ))
        return detections

    def _mock_detection(self, img: np.ndarray) -> list[BoundingBox]:
        h, w = img.shape[:2]
        return [BoundingBox(
            x1=w * 0.2, y1=h * 0.2,
            x2=w * 0.8, y2=h * 0.8,
            confidence=0.85, class_id=0, class_name="object"
        )]
