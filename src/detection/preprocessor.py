import cv2
import numpy as np
from pathlib import Path


class Preprocessor:
    def __init__(self, target_size: tuple[int, int] = (640, 640)):
        self.target_size = target_size

    def preprocess(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_resized = cv2.resize(img, self.target_size)
        img_normalized = img_resized / 255.0

        return img_normalized

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)

        enhanced = cv2.equalizeHist(gray)

        return enhanced
