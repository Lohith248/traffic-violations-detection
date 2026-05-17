#!/usr/bin/env python3
"""
One-shot export: COCO 2017 (subset) motorcycle + person → YOLO layout under datasets/raw/

  pip install fiftyone

Then merge with the rest:

  python coco_motorcycle_person_export.py
  python download_models.py --merge-only

Class names exported are motorcycle / person → remap in download_models merges to indices 0,1 already.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export COCO motorcycle+person to YOLO for datasets/raw/")
    p.add_argument(
        "--export-dir",
        type=str,
        default="./datasets/raw/coco_moto_person",
        help="Output folder (YOLO bundle with data.yaml)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "validation", "test"),
        help="COCO split to sample from",
    )
    p.add_argument("--max-samples", type=int, default=3000, help="Cap images downloaded for this subset")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out = (root / args.export_dir).resolve() if not Path(args.export_dir).is_absolute() else Path(args.export_dir)
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as e:
        raise SystemExit(
            "Missing fiftyone. Install with: pip install fiftyone\n"
            f"Original error: {e}"
        ) from e

    print(f"[INFO] Loading coco-2017 split={args.split!r}, max_samples={args.max_samples} …")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=args.split,
        label_types=["detections"],
        classes=["motorcycle", "person"],
        max_samples=int(args.max_samples),
    )

    # Overwrite previous export cleanly
    if out.exists():
        print(f"[INFO] Removing existing {out}")
        import shutil

        shutil.rmtree(out)

    dataset.export(
        export_dir=str(out),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="detections",
    )

    ds_y = out / "dataset.yaml"
    data_y = out / "data.yaml"
    if ds_y.is_file() and not data_y.is_file():
        ds_y.rename(data_y)
        print("[OK] Renamed dataset.yaml → data.yaml (merge expects data.yaml)")
    elif not data_y.exists():
        nested = next(out.rglob("data.yaml"), None)
        if nested and nested.parent != out:
            print(f"[WARN] data.yaml nested at {nested}. Copy or symlink to {data_y.name} if merge fails.")

    print(f"[OK] Exported → {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
