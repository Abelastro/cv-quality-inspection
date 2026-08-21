import pytest
from src.models.schemas import BoundingBox
from src.detection.postprocessor import Postprocessor
from src.analysis.measurements import MeasurementAnalyzer


def test_filter_by_confidence():
    detections = [
        BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9, class_id=0, class_name="obj"),
        BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.2, class_id=0, class_name="obj"),
    ]
    postprocessor = Postprocessor()
    filtered = postprocessor.filter_by_confidence(detections, 0.5)
    assert len(filtered) == 1


def test_measurement():
    det = BoundingBox(x1=10, y1=20, x2=110, y2=220, confidence=0.9, class_id=0, class_name="obj")
    analyzer = MeasurementAnalyzer()
    measurement = analyzer.measure(det)
    assert measurement.width_px == 100
    assert measurement.height_px == 200
