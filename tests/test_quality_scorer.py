import pytest
from src.models.schemas import BoundingBox, Measurement, QualityResult, DetectionResult
from src.analysis.quality_scorer import QualityScorer


def test_score_no_defects():
    detections = [
        BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9, class_id=0, class_name="object"),
    ]
    scorer = QualityScorer()
    result = scorer.score("test.jpg", detections)
    assert result.quality_score == 1.0
    assert result.passed is True
    assert result.defects_found == 0


def test_score_with_defects():
    detections = [
        BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, class_id=1, class_name="scratch"),
        BoundingBox(x1=60, y1=60, x2=100, y2=100, confidence=0.8, class_id=2, class_name="dent"),
    ]
    scorer = QualityScorer()
    result = scorer.score("test.jpg", detections)
    assert result.quality_score < 1.0
    assert result.passed is False
    assert result.defects_found == 2


def test_score_by_defect_type():
    detections = [
        BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, class_id=1, class_name="scratch"),
        BoundingBox(x1=10, y1=10, x2=60, y2=60, confidence=0.7, class_id=1, class_name="scratch"),
        BoundingBox(x1=60, y1=60, x2=100, y2=100, confidence=0.8, class_id=2, class_name="dent"),
    ]
    scorer = QualityScorer()
    result = scorer.score_by_defect_type(detections)
    assert "scratch" in result
    assert "dent" in result
    assert result["scratch"] == 0.8
    assert result["dent"] == 0.8


def test_severity_score():
    detections = [
        BoundingBox(x1=10, y1=10, x2=110, y2=110, confidence=0.9, class_id=1, class_name="scratch"),
    ]
    scorer = QualityScorer()
    severity = scorer.calculate_severity_score(detections[0], image_area=10000.0)
    assert 0.0 <= severity <= 1.0
    assert severity == pytest.approx(1.0, abs=0.01)


def test_quality_report():
    results = [
        QualityResult(
            image_path="a.jpg",
            detections=[BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, class_id=1, class_name="scratch")],
            measurements=[Measurement(width_px=50, height_px=50, area_px=2500, aspect_ratio=1.0)],
            quality_score=0.8,
            defects_found=1,
            passed=False,
        ),
        QualityResult(
            image_path="b.jpg",
            detections=[BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, class_id=0, class_name="object")],
            measurements=[Measurement(width_px=50, height_px=50, area_px=2500, aspect_ratio=1.0)],
            quality_score=1.0,
            defects_found=0,
            passed=True,
        ),
    ]
    scorer = QualityScorer()
    report = scorer.generate_quality_report(results)
    assert report["total_images"] == 2
    assert report["pass_rate"] == 0.5
    assert report["average_quality"] == pytest.approx(0.9, abs=0.01)
    assert report["total_defects"] == 1
    assert "scratch" in report["defect_type_scores"]
