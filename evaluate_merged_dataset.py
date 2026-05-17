#!/usr/bin/env python3
"""
Run Ultralytics validation on datasets/merged (real domain).

Requires:
  - datasets/merged/data.yaml with paths to train/val(/test)
  - merged images + labels under those folders

Typical setup:
  pip install -r requirements_inference.txt   # or requirements.txt
  python download_models.py                   # needs ROBOFLOW_API_KEY

Examples:
  python evaluate_merged_dataset.py --weights ./actual_weights/best.pt
  python evaluate_merged_dataset.py --weights ./actual_weights/best.pt --split val --split test
  python evaluate_merged_dataset.py --data-yaml ./datasets/merged/data.yaml --json-out domain_val_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml as yml_module


def load_class_names_from_yaml(yaml_path: Path) -> List[str]:
    cfg = yml_module.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return []


def count_gt_boxes_by_class(merged_yaml_path: Path, split: str, class_names: List[str]) -> Dict[str, Any]:
    """Count YOLO label lines per class id for one split (images/ under merged root)."""
    cfg = yml_module.safe_load(merged_yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", merged_yaml_path.parent)).resolve()
    labels_dir = root / split / "labels"
    per_id: Counter[int] = Counter()
    images_with_any = 0
    if not labels_dir.is_dir():
        return {
            "split": split,
            "label_dir": str(labels_dir),
            "images_with_labels": 0,
            "total_boxes": 0,
            "per_class": {n: 0 for n in class_names},
        }
    for p in labels_dir.glob("*.txt"):
        hit = False
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cid = int(line.split()[0])
            except ValueError:
                continue
            per_id[cid] += 1
            hit = True
        if hit:
            images_with_any += 1

    total = sum(per_id.values())
    per_human: Dict[str, int] = {}
    for i, n in enumerate(class_names):
        per_human[str(n)] = int(per_id.get(i, 0))
    for cid in sorted(k for k in per_id if k >= len(class_names)):
        per_human[f"class_{cid}"] = int(per_id[cid])

    return {
        "split": split,
        "images_with_labels": images_with_any,
        "total_boxes": total,
        "per_class_ids": dict(per_id),
        "per_class": per_human,
    }


def metrics_to_dict(metrics: Any, fallback_names: List[str]) -> Dict[str, Any]:
    """Serialize Ultralytics DetMetrics for JSON; use data.yaml names if results omit them."""
    box = metrics.box
    names = getattr(box, "names", {}) or {}
    if isinstance(names, dict) and names:
        name_list = [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, (list, tuple)) and names:
        name_list = [str(x) for x in names]
    else:
        name_list = list(fallback_names)

    maps_per_class: List[Dict[str, Any]] = []
    maps_arr = getattr(box, "maps", None)
    if maps_arr is not None:
        try:
            flat = maps_arr.flatten().tolist() if hasattr(maps_arr, "flatten") else list(maps_arr)
            for i, v in enumerate(flat):
                cls_name = name_list[i] if i < len(name_list) else str(i)
                maps_per_class.append({"class_index": i, "class": cls_name, "map50_95": float(v)})
        except Exception:
            maps_per_class = []

    out: Dict[str, Any] = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class_map50_95": maps_per_class,
        "names": name_list,
    }
    return out


def warn_if_missing_classes(gt_summary: Dict[str, Any], class_names: List[str]) -> None:
    train_pc = gt_summary.get("train", {}).get("per_class", {})
    if not train_pc or not class_names:
        return
    zeros = [n for n in class_names if int(train_pc.get(str(n), 0)) == 0]
    if zeros:
        print(
            "\n[WARN] Training split has ZERO ground-truth boxes for these declared classes:\n  "
            + ", ".join(zeros)
            + "\n  mAP for those IDs is meaningless until you merge sources that actually label them.\n",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO val/test on datasets/merged (domain evaluation).")
    p.add_argument(
        "--data-yaml",
        type=str,
        default="./datasets/merged/data.yaml",
        help="Merged dataset YAML",
    )
    p.add_argument(
        "--weights",
        type=str,
        default="./actual_weights/best.pt",
        help="Fine-tuned .pt checkpoint",
    )
    p.add_argument(
        "--split",
        action="append",
        choices=("val", "test"),
        default=[],
        help="Repeat flag: --split val --split test (default: both if test exists in yaml)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="", help="cpu, 0, 0,1, …")
    p.add_argument("--json-out", type=str, default="", help="Write combined metrics JSON")
    p.add_argument("--plots", action="store_true", help="Save PR / confusion plots under runs/")
    return p.parse_args()


def yaml_has_test(yaml_path: Path) -> bool:
    try:
        cfg = yml_module.safe_load(yaml_path.read_text(encoding="utf-8"))
        return bool(cfg.get("test"))
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    yaml_path = (root / args.data_yaml).resolve() if not Path(args.data_yaml).is_absolute() else Path(args.data_yaml)
    weights_path = (root / args.weights).resolve() if not Path(args.weights).is_absolute() else Path(args.weights)

    if not yaml_path.is_file():
        print(
            "[ERROR] Missing merged dataset YAML:\n"
            f"  Expected: {yaml_path}\n\n"
            "Build it once on any machine with Roboflow access:\n"
            "  export ROBOFLOW_API_KEY=\"…\"\n"
            "  pip install -r requirements_download.txt\n"
            "  python download_models.py\n\n"
            "Then copy the entire datasets/ folder here (or clone from your SSH GPU machine).\n",
            file=sys.stderr,
        )
        return 2

    if not weights_path.is_file():
        print(f"[ERROR] Missing weights: {weights_path}", file=sys.stderr)
        return 2

    splits: List[str]
    if args.split:
        splits = list(dict.fromkeys(args.split))
    else:
        splits = ["val"]
        if yaml_has_test(yaml_path):
            splits.append("test")

    class_names = load_class_names_from_yaml(yaml_path)
    gt_counts: Dict[str, Any] = {}
    for sp in ("train", "val", "test"):
        gt_counts[sp] = count_gt_boxes_by_class(yaml_path, sp, class_names)

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    device_kw = args.device.strip() if args.device else None

    report: Dict[str, Any] = {
        "weights": str(weights_path),
        "data_yaml": str(yaml_path),
        "imgsz": args.imgsz,
        "class_names": class_names,
        "gt_box_counts": gt_counts,
        "splits": {},
    }

    warn_if_missing_classes(gt_counts, class_names)

    for split in splits:
        print(f"\n[INFO] Running validation split={split!r} …")
        kwargs: Dict[str, Any] = {
            "data": str(yaml_path),
            "split": split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "plots": args.plots,
            "verbose": True,
        }
        if device_kw:
            kwargs["device"] = device_kw

        metrics = model.val(**kwargs)
        report["splits"][split] = metrics_to_dict(metrics, class_names)

    print("\n[GT BOX COUNTS] (training coverage — zeros here mean \"model never saw labels for that head\")")
    for sp in ("train", "val", "test"):
        block = gt_counts.get(sp, {})
        pc = block.get("per_class", {})
        print(f"  {sp}: total_boxes={block.get('total_boxes', 0)}  {json.dumps(pc)}")

    summary_out = {
        "gt_box_counts": {k: v.get("per_class", {}) for k, v in gt_counts.items()},
        "metrics": report["splits"],
    }

    print("\n[SUMMARY]")
    print(json.dumps(summary_out, indent=2))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[OK] Wrote {out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
