import pytest
import numpy as np
from src.models.schemas import BoundingBox
from src.detection.postprocessor import Postprocessor


def test_filter_by_confidence():
    detections = [
        BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9, class_id=0, class_name="obj"),
        BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.2, class_id=0, class_name="obj"),
    ]
    pp = Postprocessor()
    filtered = pp.filter_by_confidence(detections, 0.5)
    assert len(filtered) == 1
    assert filtered[0].confidence == 0.9


def test_filter_empty_detections():
    pp = Postprocessor()
    result = pp.filter_by_confidence([], 0.5)
    assert result == []


def test_nms_no_overlap():
    detections = [
        BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, class_id=0, class_name="a"),
        BoundingBox(x1=200, y1=200, x2=300, y2=300, confidence=0.8, class_id=0, class_name="a"),
    ]
    pp = Postprocessor()
    result = pp.nms(detections, iou_threshold=0.5)
    assert len(result) == 2


def test_nms_complete_overlap():
    detections = [
        BoundingBox(x1=10, y1=10, x2=100, y2=100, confidence=0.9, class_id=0, class_name="a"),
        BoundingBox(x1=10, y1=10, x2=100, y2=100, confidence=0.8, class_id=0, class_name="a"),
    ]
    pp = Postprocessor()
    result = pp.nms(detections, iou_threshold=0.5)
    assert len(result) == 1


def test_nms_partial_overlap():
    detections = [
        BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9, class_id=0, class_name="a"),
        BoundingBox(x1=50, y1=50, x2=150, y2=150, confidence=0.8, class_id=0, class_name="a"),
    ]
    pp = Postprocessor()
    result = pp.nms(detections, iou_threshold=0.3)
    assert len(result) >= 1
