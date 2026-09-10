import pytest
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock, patch
from src.models.schemas import BoundingBox, DetectionResult
from src.detection.detector import Detector


def test_detector_mock_detection():
    detector = Detector(model_path=None)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(f.name, img)
        path = f.name

    try:
        result = detector.detect(path)
        assert isinstance(result, DetectionResult)
        assert result.image_path == path
        assert len(result.detections) > 0
        assert result.processing_time_ms >= 0
    finally:
        os.unlink(path)


def test_detector_with_no_model():
    detector = Detector(model_path=None)
    assert detector.model is None


def test_detector_invalid_image():
    detector = Detector(model_path=None)
    with pytest.raises(ValueError, match="Could not read image"):
        detector.detect("/nonexistent/image.jpg")


def test_detection_result_schema():
    det = BoundingBox(
        x1=0.0, y1=0.0, x2=100.0, y2=100.0,
        confidence=0.95, class_id=0, class_name="scratch"
    )
    result = DetectionResult(
        image_path="test.jpg",
        detections=[det],
        processing_time_ms=12.5,
    )
    assert result.image_path == "test.jpg"
    assert len(result.detections) == 1
    assert result.detections[0].class_name == "scratch"
    assert result.processing_time_ms == 12.5
