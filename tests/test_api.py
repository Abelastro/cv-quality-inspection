import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
import os
import numpy as np
import cv2
from src.api.app import app, inspection_history


@pytest.fixture(autouse=True)
def clear_history():
    inspection_history.clear()
    yield
    inspection_history.clear()


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_inspect_endpoint():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img)
        path = f.name

    try:
        with open(path, "rb") as f:
            response = client.post("/inspect", files={"file": ("test.jpg", f, "image/jpeg")})
        assert response.status_code == 200
        data = response.json()
        assert "quality_score" in data
        assert "detections" in data
        assert "passed" in data
    finally:
        os.unlink(path)


def test_batch_inspect_endpoint():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    files = []
    paths = []

    for i in range(2):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            paths.append(f.name)

    try:
        file_list = []
        for p in paths:
            file_list.append(("files", ("test.jpg", open(p, "rb"), "image/jpeg")))

        response = client.post("/inspect/batch", files=file_list)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2
    finally:
        for p in paths:
            os.unlink(p)
        for _, fobj in file_list:
            fobj[1].close()


def test_statistics_endpoint():
    response = client.get("/statistics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_inspections"] == 0
    assert data["average_quality"] == 0.0


def test_history_endpoint():
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["history"] == []
