from fastapi import FastAPI, UploadFile, File
from ..detection.detector import Detector
from ..analysis.quality_scorer import QualityScorer
from ..models.schemas import QualityResult
import tempfile
import os


app = FastAPI(title="CV Quality Inspection", version="0.1.0")

detector = Detector()
scorer = QualityScorer()


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
        return result
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {"status": "healthy"}
