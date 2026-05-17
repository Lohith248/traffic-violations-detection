#!/usr/bin/env python3
"""
Count YOLO label lines per class for each split referenced in datasets/merged/data.yaml.

Usage:
  python check_merged_class_counts.py
  python check_merged_class_counts.py --data-yaml ./datasets/merged/data.yaml --splits train val
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import yaml as yml_module


def names_list(cfg: dict) -> List[str]:
    names = cfg.get("names", [])
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x))]
    raise ValueError("data.yaml missing names list")


def label_dir_for_split(merged_root: Path, cfg: dict, split_key: str) -> Path:
    rel = cfg.get(split_key)
    if not rel:
        raise KeyError(f"No '{split_key}' in yaml")
    # e.g. train/images -> merged_root/train/labels
    return merged_root / Path(rel).parent / "labels"


def count_split(labels_dir: Path, num_classes: int) -> Counter:
    ctr: Counter = Counter()
    if not labels_dir.is_dir():
        return ctr
    for lf in labels_dir.glob("*.txt"):
        for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(float(line.split()[0]))
            if 0 <= cid < num_classes:
                ctr[cid] += 1
    return ctr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class bbox counts per split from merged data.yaml.")
    p.add_argument("--data-yaml", type=str, default="./datasets/merged/data.yaml")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help='Split keys present in YAML (often "train" "val" "test")',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    yaml_path = Path(args.data_yaml)
    if not yaml_path.is_absolute():
        yaml_path = root / yaml_path
    if not yaml_path.is_file():
        print(f"[ERROR] Missing {yaml_path}", file=sys.stderr)
        return 2

    cfg = yml_module.safe_load(yaml_path.read_text(encoding="utf-8"))
    merged_root = Path(cfg.get("path", yaml_path.parent)).resolve()
    class_names = names_list(cfg)
    num_classes = len(class_names)

    bad = False
    for split in args.splits:
        if split not in cfg:
            print(f"[SKIP] No '{split}' key in yaml.")
            continue
        ld = label_dir_for_split(merged_root, cfg, split)
        ctr = count_split(ld, num_classes)
        print(f"\n{split}: labels dir = {ld}")
        for i, cls in enumerate(class_names):
            n = int(ctr.get(i, 0))
            ok = "OK" if n > 0 else "ZERO"
            print(f"  [{i}] {cls}: {n} boxes  ({ok})")
            if split == "train" and n <= 0:
                bad = True

    print()
    if bad:
        print(
            "[RESULT] Train has at least one class with zero boxes — do not rely on "
            "5-class metrics; fix merge/sources first."
        )
        return 1
    print("[RESULT] Train has nonzero boxes for every declared class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

