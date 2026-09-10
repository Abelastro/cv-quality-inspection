from concurrent.futures import ThreadPoolExecutor, as_completed
from ..detection.detector import Detector
from ..models.schemas import DetectionResult


def batch_detect(image_paths: list[str], detector: Detector) -> list[DetectionResult]:
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(detector.detect, path): path
            for path in image_paths
        }
        for future in as_completed(future_to_path):
            results.append(future.result())
    return results


def batch_detect_with_progress(
    image_paths: list[str],
    detector: Detector,
    callback=None,
) -> list[DetectionResult]:
    results = []
    total = len(image_paths)

    with ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(detector.detect, path): path
            for path in image_paths
        }
        for i, future in enumerate(as_completed(future_to_path)):
            result = future.result()
            results.append(result)
            if callback:
                callback(i + 1, total, result)

    return results
