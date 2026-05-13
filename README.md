# Traffic Violation Detection Submission (AID 728)

## Directory structure (required)

```text
.
├─ solution.py
├─ train.py
├─ test_single.py
├─ download_models.py
├─ requirements.txt
├─ README.md
├─ datasets/
│  ├─ raw/                 # auto-created by download_models.py
│  └─ merged/
│     └─ data.yaml         # merged YOLO dataset for training
└─ models/
   ├─ yolov8s.pt
   ├─ yolov8s_traffic.pt   # produced by train.py
   ├─ paddle_det/
   │  ├─ inference.pdmodel
   │  └─ inference.pdiparams
   ├─ paddle_rec/
   │  ├─ inference.pdmodel
   │  └─ inference.pdiparams
   └─ paddle_cls/
      ├─ inference.pdmodel
      └─ inference.pdiparams
```

## One-time setup (models + datasets)

Set Roboflow API key, then run once:

```powershell
$env:ROBOFLOW_API_KEY="YOUR_API_KEY"
python download_models.py
```

`download_models.py` will:
1. Download `yolov8s.pt` into `./models/`.
2. Download PaddleOCR det/rec/cls into `./models/paddle_det`, `./models/paddle_rec`, `./models/paddle_cls`.
3. Download and merge the required Roboflow datasets into `./datasets/merged/data.yaml`.
4. Print model file sizes and total model size.

## Train

```bash
python train.py --data-yaml ./datasets/merged/data.yaml --model-dir ./models --epochs 50 --imgsz 640
```

Best checkpoint is copied to:

```text
./models/yolov8s_traffic.pt
```

## Test single image

```bash
python test_single.py <image_path> [model_dir]
```

Example:

```bash
python test_single.py ./sample.jpg ./models
```

## Evaluator loading behavior

The evaluator dynamically imports `TrafficViolationDetector` from `solution.py` and uses:

```python
model = TrafficViolationDetector("./models")
output = model.predict(image_path)
```

## Model size breakdown (target)

| File/Folder | Size (approx.) |
|---|---:|
| `models/yolov8s.pt` | ~21.5 MB |
| `models/yolov8s_traffic.pt` | ~21.5 MB |
| `models/paddle_det/` | ~4.7 MB |
| `models/paddle_rec/` | ~9.8 MB |
| `models/paddle_cls/` | ~1.4 MB |
| **Total** | **~58.9 MB** |

Total model size is well under the 250 MB limit. Use the size report printed by `download_models.py` for exact local values.
