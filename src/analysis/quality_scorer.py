from ..models.schemas import BoundingBox, Measurement, QualityResult, DetectionResult
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

    def score_by_defect_type(
        self, detections: list[BoundingBox]
    ) -> dict[str, float]:
        type_scores: dict[str, list[float]] = {}
        for det in detections:
            if det.class_name != "object":
                type_scores.setdefault(det.class_name, []).append(det.confidence)

        scores = {}
        for class_name, confidences in type_scores.items():
            avg = sum(confidences) / len(confidences)
            scores[class_name] = round(avg, 4)
        return scores

    def calculate_severity_score(
        self,
        detection: BoundingBox,
        image_area: float,
    ) -> float:
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1
        area = width * height

        if image_area <= 0:
            return 0.0

        ratio = area / image_area
        return round(min(ratio, 1.0), 4)

    def generate_quality_report(
        self, results: list[QualityResult]
    ) -> dict:
        if not results:
            return {
                "total_images": 0,
                "average_quality": 0.0,
                "pass_rate": 0.0,
                "total_defects": 0,
                "defect_type_scores": {},
            }

        total = len(results)
        avg_quality = sum(r.quality_score for r in results) / total
        passed = sum(1 for r in results if r.passed)
        total_defects = sum(r.defects_found for r in results)

        combined_detections = []
        for r in results:
            combined_detections.extend(r.detections)

        type_scores = self.score_by_defect_type(combined_detections)

        return {
            "total_images": total,
            "average_quality": round(avg_quality, 4),
            "pass_rate": round(passed / total, 4),
            "total_defects": total_defects,
            "defect_type_scores": type_scores,
        }
