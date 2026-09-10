import pytest
from src.models.schemas import BoundingBox, Measurement, QualityResult
from src.analysis.statistics import (
    calculate_defect_distribution,
    calculate_average_quality,
    detect_quality_trends,
)


def test_defect_distribution():
    results = [
        QualityResult(
            image_path="a.jpg",
            detections=[
                BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, class_id=1, class_name="scratch"),
                BoundingBox(x1=60, y1=60, x2=100, y2=100, confidence=0.8, class_id=2, class_name="dent"),
            ],
            measurements=[Measurement(width_px=50, height_px=50, area_px=2500, aspect_ratio=1.0)],
            quality_score=0.6,
            defects_found=2,
            passed=False,
        ),
        QualityResult(
            image_path="b.jpg",
            detections=[
                BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, class_id=1, class_name="scratch"),
            ],
            measurements=[Measurement(width_px=50, height_px=50, area_px=2500, aspect_ratio=1.0)],
            quality_score=0.8,
            defects_found=1,
            passed=False,
        ),
    ]
    dist = calculate_defect_distribution(results)
    assert dist["scratch"] == 2
    assert dist["dent"] == 1


def test_average_quality():
    results = [
        QualityResult(
            image_path="a.jpg", detections=[], measurements=[],
            quality_score=0.8, defects_found=0, passed=True,
        ),
        QualityResult(
            image_path="b.jpg", detections=[], measurements=[],
            quality_score=1.0, defects_found=0, passed=True,
        ),
    ]
    avg = calculate_average_quality(results)
    assert avg == pytest.approx(0.9, abs=0.01)


def test_average_quality_empty():
    assert calculate_average_quality([]) == 0.0


def test_quality_trends():
    results_with_ts = [
        (1.0, QualityResult(
            image_path="a.jpg", detections=[], measurements=[],
            quality_score=0.6, defects_found=0, passed=True,
        )),
        (2.0, QualityResult(
            image_path="b.jpg", detections=[], measurements=[],
            quality_score=0.7, defects_found=0, passed=True,
        )),
        (3.0, QualityResult(
            image_path="c.jpg", detections=[], measurements=[],
            quality_score=0.8, defects_found=0, passed=True,
        )),
        (4.0, QualityResult(
            image_path="d.jpg", detections=[], measurements=[],
            quality_score=0.9, defects_found=0, passed=True,
        )),
    ]
    trend = detect_quality_trends(results_with_ts)
    assert trend["trend"] == "improving"
    assert trend["data_points"] == 4


def test_quality_trends_insufficient():
    results_with_ts = [
        (1.0, QualityResult(
            image_path="a.jpg", detections=[], measurements=[],
            quality_score=0.8, defects_found=0, passed=True,
        )),
    ]
    trend = detect_quality_trends(results_with_ts)
    assert trend["trend"] == "insufficient_data"
