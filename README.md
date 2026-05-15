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

## Google Colab training (YOLOv8m)

Use this when you want to clone the repo and train directly in Colab.

```bash
!git clone <YOUR_REPO_URL>
%cd traffic-violations-detection
!pip install -U pip
!pip install ultralytics==8.2.103 albumentations==1.3.1 opencv-python-headless==4.10.0.84 PyYAML==6.0.2 numpy==1.23.5 roboflow==1.1.37
```

If `./datasets/merged/data.yaml` is not already in your repo, download and merge datasets:

```bash
import os
os.environ["ROBOFLOW_API_KEY"] = "YOUR_API_KEY"
!python download_models.py
```

Train YOLOv8m:

```bash
!python train.py \
  --data-yaml ./datasets/merged/data.yaml \
  --model-dir ./models \
  --base-weights yolov8m.pt \
  --output-weights yolov8m_traffic.pt \
  --rebalance-train \
  --max-repeats 3 \
  --augment-offline \
  --aug-copies-per-image 1 \
  --epochs 50 \
  --imgsz 640 \
  --device 0
```

`train.py` now auto-downloads missing base weights (`yolov8m.pt`, `yolov8s.pt`) into `./models`.

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
