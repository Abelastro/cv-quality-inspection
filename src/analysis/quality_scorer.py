from ..models.schemas import BoundingBox, Measurement, QualityResult
from .measurements import MeasurementAnalyzer


class QualityScorer:
    def __init__(self, defect_threshold: float = 0.3):
        self.defect_threshold = defect_threshold
        self.analyzer = MeasurementAnalyzer()

    def score(
        self,
        image_path: str,
        detections: list[BoundingBox],
        processing_time_ms: float = 0,
    ) -> QualityResult:
        measurements = self.analyzer.measure_all(detections)

        defects = [d for d in detections if d.class_name != "object"]
        defects_found = len(defects)

        if defects_found == 0:
            quality_score = 1.0
        else:
            quality_score = max(0, 1.0 - (defects_found * 0.2))

        passed = defects_found == 0

        return QualityResult(
            image_path=image_path,
            detections=detections,
            measurements=measurements,
            quality_score=quality_score,
            defects_found=defects_found,
            passed=passed,
        )
