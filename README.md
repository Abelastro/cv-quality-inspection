# CV Quality Inspection

Computer vision system for **object detection**, **defect analysis**, and **quality scoring**. YOLO-based detection with OpenCV preprocessing and FastAPI inference.

## Why I Built This

This project demonstrates practical computer vision engineering — going beyond simple image classification to build a system that could exist in an industrial quality inspection environment.

## Architecture

```
Input Image
    ↓
Preprocessing (OpenCV)
    ↓
Object Detection (YOLO)
    ↓
Defect Analysis
    ↓
Quality Scoring
    ↓
Structured Output
    ↓
API Response
```

## Features

- **YOLO object detection** for identifying items/defects
- **OpenCV preprocessing** pipeline
- **Defect measurement** and analysis
- **Quality scoring** system
- **FastAPI REST API**
- **Docker containerization**
- **Result visualization**

## Tech Stack

- Python 3.10+
- OpenCV
- YOLO (Ultralytics)
- FastAPI
- NumPy
- Docker

## Project Structure

```
cv-quality-inspection/
├── src/
│   ├── detection/
│   │   ├── detector.py      # YOLO detection
│   │   ├── preprocessor.py  # Image preprocessing
│   │   └── postprocessor.py # Result processing
│   ├── analysis/
│   │   ├── measurements.py  # Size/shape analysis
│   │   └── quality_scorer.py
│   ├── models/
│   │   └── schemas.py
│   ├── api/
│   │   └── app.py
│   └── utils/
│       ├── config.py
│       └── visualization.py
├── models/weights/
├── data/samples/
├── tests/
├── docs/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Installation

```bash
git clone https://github.com/Abelastro/cv-quality-inspection.git
cd cv-quality-inspection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Detect Objects

```python
from src.detection.detector import Detector

detector = Detector()
results = detector.detect("path/to/image.jpg")
print(results)
```

### Analyze Quality

```python
from src.analysis.quality_scorer import QualityScorer

scorer = QualityScorer()
score = scorer.score(detections)
print(f"Quality Score: {score}")
```

### API

```bash
python -m src.main
# POST /inspect with image file
```

## Limitations

- Prototype system (not production-ready)
- Requires pre-trained YOLO weights
- Basic defect analysis implementation

## Future Improvements

- Add custom model training
- Implement tracking across frames
- Add reporting dashboard
- Support video streams
