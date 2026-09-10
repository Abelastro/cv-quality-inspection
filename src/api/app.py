from fastapi import FastAPI, UploadFile, File
from ..detection.detector import Detector
from ..analysis.quality_scorer import QualityScorer
from ..analysis.statistics import (
    calculate_defect_distribution,
    calculate_average_quality,
)
from ..models.schemas import QualityResult
import tempfile
import os

app = FastAPI(title="CV Quality Inspection", version="0.1.0")

detector = Detector()
scorer = QualityScorer()

inspection_history: list[QualityResult] = []


@app.post("/inspect", response_model=QualityResult)
async def inspect(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        detection = detector.detect(tmp_path)
        result = scorer.score(
            image_path=file.filename,
            detections=detection.detections,
            processing_time_ms=detection.processing_time_ms,
        )
        inspection_history.append(result)
        return result
    finally:
        os.unlink(tmp_path)


@app.post("/inspect/batch")
async def inspect_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            detection = detector.detect(tmp_path)
            result = scorer.score(
                image_path=file.filename,
                detections=detection.detections,
                processing_time_ms=detection.processing_time_ms,
            )
            inspection_history.append(result)
            results.append(result)
        finally:
            os.unlink(tmp_path)

    return {"results": results, "total": len(results)}


@app.get("/statistics")
async def statistics():
    if not inspection_history:
        return {
            "total_inspections": 0,
            "average_quality": 0.0,
            "defect_distribution": {},
        }

    return {
        "total_inspections": len(inspection_history),
        "average_quality": calculate_average_quality(inspection_history),
        "defect_distribution": calculate_defect_distribution(inspection_history),
    }


@app.get("/history")
async def history():
    return {"history": inspection_history, "total": len(inspection_history)}


@app.get("/health")
async def health():
    return {"status": "healthy"}
