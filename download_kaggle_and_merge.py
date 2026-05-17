#!/usr/bin/env python3
"""
One-shot: Download ALL datasets (Roboflow + Kaggle K1-K8), convert, merge, verify.

Usage:
  python download_kaggle_and_merge.py --roboflow-key YOUR_KEY --kaggle-username USER --kaggle-key KEY

Or set env vars:
  export ROBOFLOW_API_KEY=xxx KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx
  python download_kaggle_and_merge.py
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

KAGGLE_DATASETS = [
    ("aneesarom/rider-with-helmet-without-helmet-number-plate", "datasets/raw/kaggle_aneesarom"),
    ("rishabhzen/on-vehicle-helmet-detection", "datasets/raw/kaggle_rishabhzen"),
    ("meliodassourav/traffic-violation-dataset-v3", "datasets/raw/kaggle_meliodassourav"),
    ("kedarsai/indian-license-plates-with-labels", "datasets/raw/kaggle_kedarsai"),
    ("guisahanes/traffic-violation-detection-dataset", "datasets/raw/kaggle_guisahanes"),
]

def run(cmd, check=True):
    print(f"\n{'='*60}\n[CMD] {cmd}\n{'='*60}")
    return subprocess.run(cmd, shell=True, check=check).returncode

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roboflow-key", default=os.getenv("ROBOFLOW_API_KEY", ""))
    p.add_argument("--kaggle-username", default=os.getenv("KAGGLE_USERNAME", ""))
    p.add_argument("--kaggle-key", default=os.getenv("KAGGLE_KEY", ""))
    p.add_argument("--skip-kaggle", action="store_true")
    p.add_argument("--skip-roboflow", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    print("="*60 + "\n  TRAFFIC VIOLATION DETECTION - FULL DATASET PIPELINE\n" + "="*60)

    if not args.merge_only:
        run(f"{sys.executable} -m pip install -q kaggle roboflow pyyaml")

        # Roboflow
        if not args.skip_roboflow and args.roboflow_key:
            os.environ["ROBOFLOW_API_KEY"] = args.roboflow_key
            run(f"{sys.executable} download_models.py --no-paddle --no-yolov8-base", check=False)

        # Kaggle K1-K8
        if not args.skip_kaggle and args.kaggle_username and args.kaggle_key:
            os.environ["KAGGLE_USERNAME"] = args.kaggle_username
            os.environ["KAGGLE_KEY"] = args.kaggle_key
            for slug, dest in KAGGLE_DATASETS:
                if Path(dest).exists() and list(Path(dest).rglob("*.jpg")):
                    print(f"[SKIP] {dest} already has images")
                    continue
                run(f"kaggle datasets download -d {slug} -p {dest} --unzip", check=False)

        # Convert Kaggle to YOLO
        if any(Path("datasets/raw").glob("kaggle_*")):
            run(f"{sys.executable} convert_kaggle_to_yolo.py")

    # Merge all
    print("\n[STEP] Merging ALL datasets...")
    run(f"{sys.executable} download_models.py --merge-only --allow-incomplete-merge")

    # Verify
    print("\n[VERIFY] Class distribution:")
    run(f"{sys.executable} check_merged_class_counts.py")

    print("\n" + "="*60 + "\n  DONE! Next: python train.py --base-weights yolo26s.pt\n" + "="*60)

if __name__ == "__main__":
    main()
