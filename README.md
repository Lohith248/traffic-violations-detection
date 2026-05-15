# Traffic Violation Detection (SSH GPU Workflow)

This repo is now configured for **SSH GPU training first** (no Colab required).

## 1) Required folder layout

```text
traffic-violations-detection/
├─ solution.py
├─ train.py
├─ test_single.py
├─ download_models.py
├─ requirements.txt              # training environment
├─ requirements_download.txt     # dataset/model download helper deps
├─ requirements_inference.txt    # inference/evaluator environment
├─ README.md
├─ datasets/
│  ├─ raw/                       # auto-created by download_models.py
│  └─ merged/
│     ├─ data.yaml
│     ├─ train/images, train/labels
│     ├─ val/images, val/labels
│     └─ test/images, test/labels
└─ models/
   ├─ yolov8s.pt
   ├─ yolov8m.pt                 # auto-downloaded when needed
   ├─ yolov8s_traffic.pt
   ├─ yolov8m_traffic.pt
   ├─ paddle_det/
   ├─ paddle_rec/
   └─ paddle_cls/
```

## 2) Create clean SSH env (Python 3.10 recommended)

```bash
conda create -n tvd310 python=3.10 -y
conda activate tvd310
python -m pip install --upgrade pip setuptools wheel
```

## 3) Install training dependencies (conflict-free)

```bash
pip install -r requirements.txt
```

Use this for **training only**.

## 4) Download/merge datasets and base models (one-time)

Install downloader deps (separate on purpose):

```bash
pip install -r requirements_download.txt
```

Set key and run:

```bash
export ROBOFLOW_API_KEY="YOUR_API_KEY"
python download_models.py
```

`download_models.py` now:
1. Downloads YOLO base weights and Paddle OCR assets.
2. Merges Roboflow datasets into 5 target classes.
3. Sanitizes YOLO boxes.
4. Trims excessive background-only images.
5. Prints split-wise class counts.
6. Fails early if train split misses any target class.

## 5) Train YOLOv8m on SSH GPU

```bash
python train.py \
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

Notes:
- `train.py` auto-downloads missing base weights (`yolov8s.pt` / `yolov8m.pt`) into `./models`.
- `train.py` audits class coverage before actual training.

## 6) Quick local test

```bash
python test_single.py /path/to/sample.jpg ./models
```

## 7) Inference/evaluator dependencies (separate env recommended)

If you need to run full OCR inference (`solution.py`) with PaddleOCR:

```bash
pip install -r requirements_inference.txt
```

This split avoids the resolver conflicts you were hitting during GPU training setup.
