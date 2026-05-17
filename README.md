# Traffic Rule Violation Detection

**Team 7** — AID 728 Course Project  
Ravi Abhinav (BT2024239) · Pranav (BT2024221) · Lohith (BT2024248)

An automated computer vision system that detects motorcycle traffic violations (helmet non-compliance and triple riding) from street scene images and extracts the license plate of offending vehicles via OCR.

## System Overview

The pipeline uses a **two-stage architecture**:

1. **YOLO26s** — Detects motorcycles, persons, helmets, no-helmets, and license plates (NMS-free, 19.4 MB, ~133ms on CPU).
2. **PaddleOCR v4** — Extracts license plate text from violating vehicles with dynamic preprocessing and time-capped execution.

**Key Results:** Overall **mAP50 = 0.916** on the test split, with inference under 1 second per image on CPU.

## Submission Structure

```
<ROLL_NUMBER>/
├── solution.py             # Main class: TrafficViolationDetector
├── models/                 # Model weights (≤ 250 MB total)
│   ├── yolo26s_traffic.pt  # Fine-tuned YOLO26s (5 classes, 19.4 MB)
│   ├── paddle_det/         # PaddleOCR text detection model
│   ├── paddle_rec/         # PaddleOCR text recognition model
│   └── paddle_cls/         # PaddleOCR angle classification model
├── requirements.txt        # All Python dependencies
└── README.md               # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Inference
```python
from solution import TrafficViolationDetector

detector = TrafficViolationDetector(model_dir="./models")
result = detector.predict("path/to/street_image.jpg")
print(result)
```

### 3. Output Format
```json
{
  "violations": [
    {
      "num_riders": 3,
      "helmet_violations": 2,
      "license_plate": "KA01AB1234"
    }
  ]
}
```
- `num_riders`: Total riders detected on the motorcycle.
- `helmet_violations`: Number of riders without helmets.
- `license_plate`: OCR-extracted plate text (empty string if unreadable).
- Only violating vehicles are reported. Compliant vehicles return an empty list.

## How It Works

1. **Detection:** YOLO26s localizes all motorcycles, persons, helmets, and license plates.
2. **Rider Association:** Persons are linked to motorcycles via IoU overlap (threshold ≥ 0.15).
3. **Helmet Check:** The top 35% of each rider's bounding box is checked for a helmet centroid.
4. **Violation Flagging:** A motorcycle is flagged if `helmet_violations > 0` OR `num_riders > 2`.
5. **OCR:** PaddleOCR extracts the plate text with up to 4 attempts, soft-capped at 4.3 seconds.

## Model Performance (Test Split)

| Class         | AP50  | AP50-95 |
|---------------|-------|---------|
| motorcycle    | 0.924 | 0.648   |
| person        | 0.968 | 0.815   |
| helmet        | 0.908 | 0.577   |
| no_helmet     | 0.836 | 0.494   |
| license_plate | 0.946 | 0.678   |
| **Overall**   | **0.916** | **0.643** |

## Constraints Compliance

| Constraint            | Limit      | Actual     |
|-----------------------|------------|------------|
| Total model size      | ≤ 250 MB   | ~65 MB ✅  |
| Large models (>1B)    | Prohibited | Not used ✅|
| Internet access       | Disabled   | Offline ✅ |
| Inference time        | ~5 seconds | ~750ms ✅  |

## Dataset

Trained on 4,291 images aggregated from 11 sources (Roboflow + Kaggle), unified into 5 classes with 80+ alias mappings resolved programmatically.

## Tech Stack

- Python 3.10+, PyTorch, CUDA
- Ultralytics 8.4.51 (YOLO26s)
- PaddleOCR v4
- OpenCV, Albumentations, NumPy
