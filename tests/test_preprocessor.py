import pytest
import numpy as np
import tempfile
import os
import cv2
from src.detection.preprocessor import Preprocessor


def _create_test_image(tmp_dir, name="test.png", size=(200, 200)):
    path = os.path.join(tmp_dir, name)
    img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def test_preprocess_resize():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _create_test_image(tmp_dir, size=(300, 200))
        pp = Preprocessor(target_size=(100, 100))
        result = pp.preprocess(path)
        assert result.shape == (100, 100, 3)


def test_preprocess_normalize():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _create_test_image(tmp_dir)
        pp = Preprocessor(target_size=(64, 64))
        result = pp.preprocess(path)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


def test_enhance_grayscale():
    pp = Preprocessor()
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    enhanced = pp.enhance(img / 255.0)
    assert enhanced.shape == (100, 100)
    assert enhanced.dtype == np.uint8


def test_enhance_color():
    pp = Preprocessor()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    enhanced = pp.enhance(img / 255.0)
    assert enhanced.shape == (100, 100)
    assert enhanced.dtype == np.uint8


def test_denoise():
    pp = Preprocessor()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    denoised = pp.denoise(img)
    assert denoised.shape == img.shape
    assert denoised.dtype == np.uint8


def test_sharpen():
    pp = Preprocessor()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    sharpened = pp.sharpen(img)
    assert sharpened.shape == img.shape
    assert sharpened.dtype == np.uint8


def test_adjust_contrast():
    pp = Preprocessor()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    adjusted = pp.adjust_contrast(img, factor=2.0)
    assert adjusted.shape == img.shape
    assert adjusted.dtype == np.uint8
