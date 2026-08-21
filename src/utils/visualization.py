import cv2
import numpy as np
from ..models.schemas import BoundingBox


def draw_detections(image: np.ndarray, detections: list[BoundingBox]) -> np.ndarray:
    vis = image.copy()

    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        color = (0, 255, 0) if det.class_name == "object" else (0, 0, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name}: {det.confidence:.2f}"
        cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis
