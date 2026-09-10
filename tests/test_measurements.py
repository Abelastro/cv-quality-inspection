import pytest
from src.models.schemas import BoundingBox
from src.analysis.measurements import MeasurementAnalyzer


def test_measure_square():
    det = BoundingBox(x1=10, y1=10, x2=110, y2=110, confidence=0.9, class_id=0, class_name="obj")
    analyzer = MeasurementAnalyzer()
    m = analyzer.measure(det)
    assert m.width_px == 100
    assert m.height_px == 100
    assert m.area_px == 10000
    assert m.aspect_ratio == 1.0


def test_measure_rectangle():
    det = BoundingBox(x1=0, y1=0, x2=200, y2=100, confidence=0.9, class_id=0, class_name="obj")
    analyzer = MeasurementAnalyzer()
    m = analyzer.measure(det)
    assert m.width_px == 200
    assert m.height_px == 100
    assert m.area_px == 20000
    assert m.aspect_ratio == 2.0


def test_measure_zero_height():
    det = BoundingBox(x1=10, y1=50, x2=110, y2=50, confidence=0.9, class_id=0, class_name="obj")
    analyzer = MeasurementAnalyzer()
    m = analyzer.measure(det)
    assert m.height_px == 0
    assert m.aspect_ratio == 0


def test_measure_all():
    detections = [
        BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, class_id=0, class_name="a"),
        BoundingBox(x1=10, y1=10, x2=60, y2=60, confidence=0.8, class_id=0, class_name="b"),
    ]
    analyzer = MeasurementAnalyzer()
    results = analyzer.measure_all(detections)
    assert len(results) == 2
    assert results[0].area_px == 2500
    assert results[1].area_px == 2500
