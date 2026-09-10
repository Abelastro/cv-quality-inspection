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

    def denoise(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.copy()
        return cv2.GaussianBlur(img, (5, 5), 0)

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.copy()
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    def adjust_contrast(self, image: np.ndarray, factor: float = 2.0) -> np.ndarray:
        if image.dtype != np.uint8:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.copy()

        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8, 8))
            return clahe.apply(img)
