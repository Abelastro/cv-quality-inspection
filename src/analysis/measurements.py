import numpy as np
from ..models.schemas import BoundingBox, Measurement


class MeasurementAnalyzer:
    def measure(self, detection: BoundingBox) -> Measurement:
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0

        return Measurement(
            width_px=width,
            height_px=height,
            area_px=area,
            aspect_ratio=aspect_ratio,
        )

    def measure_all(self, detections: list[BoundingBox]) -> list[Measurement]:
        return [self.measure(d) for d in detections]
