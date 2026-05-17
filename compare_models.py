#!/usr/bin/env python3
"""
compare_models.py — Evaluate yolo26s_traffic.pt vs yolo11s_traffic.pt on the test split
and print a side-by-side comparison of per-class AP50 and overall mAP50/mAP50-95.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any

from ultralytics import YOLO

DATA_YAML   = Path("./datasets/merged/data.yaml")
MODEL_DIR   = Path("./models")
CLASS_NAMES = ["motorcycle", "person", "helmet", "no_helmet", "license_plate"]

MODELS = {
    "YOLO26s": MODEL_DIR / "yolo26s_traffic.pt",
    "YOLO11s": MODEL_DIR / "yolo11s_traffic.pt",
}

def run_val(model_path: Path) -> Dict[str, Any]:
    model = YOLO(str(model_path))
    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        device="0",
        verbose=False,
        save=False,
        plots=False,
    )
    # Pull per-class AP50
    ap50_per_class: List[float] = results.box.ap50.tolist()   # shape: [num_classes]
    map50: float    = float(results.box.map50)
    map5095: float  = float(results.box.map)
    return {
        "ap50_per_class": ap50_per_class,
        "map50": map50,
        "map5095": map5095,
    }

def print_table(all_results: Dict[str, Dict[str, Any]]) -> None:
    col_w  = 14
    names  = list(all_results.keys())

    # Header
    header_cells = ["Class".ljust(18)] + [n.center(col_w) for n in names]
    print("\n" + "=" * (18 + col_w * len(names)))
    print("  MODEL COMPARISON — Test Split  (AP50 per class, mAP50, mAP50-95)")
    print("=" * (18 + col_w * len(names)))
    print("  " + " | ".join(header_cells))
    print("  " + "-" * (16 + (col_w + 3) * len(names)))

    # Per-class rows
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        row = [cls_name.ljust(18)]
        for name in names:
            ap = all_results[name]["ap50_per_class"][cls_idx]
            row.append(f"{ap:.4f}".center(col_w))
        print("  " + " | ".join(row))

    print("  " + "-" * (16 + (col_w + 3) * len(names)))

    # mAP50 row
    row = ["Overall mAP50".ljust(18)]
    for name in names:
        row.append(f"{all_results[name]['map50']:.4f}".center(col_w))
    print("  " + " | ".join(row))

    # mAP50-95 row
    row = ["mAP50-95".ljust(18)]
    for name in names:
        row.append(f"{all_results[name]['map5095']:.4f}".center(col_w))
    print("  " + " | ".join(row))

    print("=" * (18 + col_w * len(names)))

    # Winner
    winner = max(all_results, key=lambda n: all_results[n]["map50"])
    print(f"\n  🏆 Winner (mAP50): {winner}  ({all_results[winner]['map50']:.4f})")
    print()

def main() -> None:
    all_results: Dict[str, Dict[str, Any]] = {}
    for label, model_path in MODELS.items():
        if not model_path.exists():
            print(f"[SKIP] {label}: {model_path} not found.")
            continue
        print(f"\n[EVAL] {label}  →  {model_path}")
        all_results[label] = run_val(model_path)
        m50 = all_results[label]["map50"]
        m5095 = all_results[label]["map5095"]
        print(f"       mAP50={m50:.4f}  mAP50-95={m5095:.4f}")

    if len(all_results) < 2:
        print("[WARN] Only one model evaluated — nothing to compare.")
    else:
        print_table(all_results)

if __name__ == "__main__":
    main()
