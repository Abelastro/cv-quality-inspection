from collections import Counter
from ..models.schemas import QualityResult


def calculate_defect_distribution(results: list[QualityResult]) -> dict[str, int]:
    counter: Counter = Counter()
    for result in results:
        for det in result.detections:
            if det.class_name != "object":
                counter[det.class_name] += 1
    return dict(counter)


def calculate_average_quality(results: list[QualityResult]) -> float:
    if not results:
        return 0.0
    return sum(r.quality_score for r in results) / len(results)


def detect_quality_trends(
    results_with_timestamps: list[tuple[float, QualityResult]],
) -> dict:
    if len(results_with_timestamps) < 2:
        return {"trend": "insufficient_data", "data_points": len(results_with_timestamps)}

    sorted_data = sorted(results_with_timestamps, key=lambda x: x[0])
    scores = [r.quality_score for _, r in sorted_data]
    n = len(scores)

    first_half = scores[: n // 2]
    second_half = scores[n // 2 :]

    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)

    diff = second_avg - first_avg

    if diff > 0.05:
        trend = "improving"
    elif diff < -0.05:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "data_points": n,
        "first_half_avg": round(first_avg, 4),
        "second_half_avg": round(second_avg, 4),
        "overall_avg": round(sum(scores) / n, 4),
    }
