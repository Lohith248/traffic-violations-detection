# Traffic Violation Detection (SSH GPU Workflow)

This repo is configured for **SSH GPU training first** (no Colab required).

## 1) Required folder layout

```text
traffic-violations-detection/
├─ solution.py
├─ train.py
├─ export_weights.py
├─ compare_yolo_checkpoints.py
├─ evaluate_merged_dataset.py     # domain mAP on datasets/merged val (+ test)
├─ run_inference_cases.py          # optional web-download smoke tests
├─ test_single.py
├─ download_models.py
├─ requirements.txt / requirements_download.txt / requirements_inference.txt
├─ README.md
├─ actual_weights/              # primary: best.pt + last.pt from your training run
├─ h/                           # optional older mirrors
├─ datasets/
│  └─ merged/ …
└─ models/                      # Paddle + optional yolov8*_traffic.pt copies
```

## 2) Create clean SSH env (Python 3.10 recommended)

```bash
conda create -n tvd310 python=3.10 -y
conda activate tvd310
python -m pip install --upgrade pip setuptools wheel
```

## 3) Install training dependencies

```bash
pip install -r requirements.txt
```

## 4) Download/merge datasets and base models (one-time)

```bash
pip install -r requirements_download.txt
export ROBOFLOW_API_KEY="YOUR_API_KEY"
python download_models.py
```

## 5) Train and publish weights

Copy your remote `best.pt` / `last.pt` into **`./actual_weights/`** (flat: `actual_weights/best.pt`, not a nested `actual_weights/actual_weights/` folder).

Optional copy into `models/` for course submission bundles:

```bash
python export_weights.py ./actual_weights/best.pt \
  --to-models ./models/yolov8s_traffic.pt \
  --strict
```

Training on the GPU machine:

```bash
python train.py \
  --data-yaml ./datasets/merged/data.yaml \
  --model-dir ./models \
  --base-weights yolov8s.pt \
  --output-weights yolov8s_traffic.pt \
  --export-h ./actual_weights/best.pt \
  ...
```

## 6) Quick local test

**YOLO load order:** `TRAFFIC_YOLO_WEIGHTS` → `./actual_weights/best.pt` → `./actual_weights/last.pt` → `./models/*_traffic.pt` → `./h/*.pt` → `./models/yolov8*.pt`.

```bash
python test_single.py /path/to/sample.jpg ./models
```

The script prints which **YOLO checkpoint** file was loaded.

```bash
python compare_yolo_checkpoints.py
```

## 6b) Domain evaluation (merged **val** / **test**)

**One command (downloads dataset if missing; needs API key unless you skip):**

```bash
python setup_and_eval.py
```

Windows PowerShell first: `$env:ROBOFLOW_API_KEY='…'`. Already have **`datasets/merged`**? `python setup_and_eval.py --skip-download`.

Use **`datasets/merged`** from `download_models.py` (or copy from your SSH GPU machine)—not random web images—for real mAP.

```bash
pip install -r requirements_inference.txt
python evaluate_merged_dataset.py \
  --weights ./actual_weights/best.pt \
  --data-yaml ./datasets/merged/data.yaml \
  --split val \
  --split test \
  --json-out domain_val_report.json
```

If `data.yaml` contains a `test:` split, you can omit all `--split` flags and the script runs **val** then **test**. Use `--device 0` on GPU or `--batch 4` / `--device cpu` on a laptop.

Equivalent checkpoint timing + val:

```bash
python compare_yolo_checkpoints.py \
  --weights ./actual_weights/best.pt \
  --data-yaml ./datasets/merged/data.yaml
```

## 7) Inference / OCR environment

```bash
pip install -r requirements_inference.txt
```

## FAQ

**Wrong class count (80 vs 5)?**  
The `.pt` file on disk must be your **fine-tuned** run. COCO base weights always show **nc=80**. Fine-tuned traffic weights show **5 classes** in this order: `motorcycle`, `person`, `helmet`, `no_helmet`, `license_plate`.

Verify:

```bash
python export_weights.py ./actual_weights/best.pt --strict
```
