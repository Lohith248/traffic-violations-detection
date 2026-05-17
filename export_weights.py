from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

_EXPECTED_CLASSES = ("motorcycle", "person", "helmet", "no_helmet", "license_plate")


def _load_names(weights: Path) -> Tuple[int, List[str]]:
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        raw = getattr(model, "names", {})
        if isinstance(raw, dict):
            ordered = [str(raw[int(i)]) for i in sorted(raw.keys())]
        else:
            ordered = [str(x) for x in raw]
        return len(ordered), ordered
    except Exception as exc:
        raise RuntimeError(f"Could not inspect {weights}: {exc}") from exc


def _verify_task_head(nc: int, names_lower: List[str]) -> Tuple[str, bool]:
    if nc == len(_EXPECTED_CLASSES):
        canon = tuple(n.strip().lower() for n in names_lower[: len(_EXPECTED_CLASSES)])
        if canon == tuple(_EXPECTED_CLASSES):
            return ("Aligned 5-class traffic head.", True)
        return (
            f"5 classes present but unusual names/order: {list(names_lower[:5])} "
            f"(expect {list(_EXPECTED_CLASSES)}). Inference may still work if labels map correctly.",
            True,
        )
    if nc == 80 and len(names_lower) >= 4 and names_lower[:4] == ["person", "bicycle", "car", "motorcycle"]:
        return (
            "COCO pretrained (80 classes). Not your merged traffic detector — "
            "train on merged data then export via this script.",
            False,
        )
    return (
        f"Checkpoint has nc={nc} (expected {len(_EXPECTED_CLASSES)} for this project). Confirm before shipping.",
        False,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Copy Ultralytics-trained weights to submission paths and verify class head.",
    )
    p.add_argument("checkpoint", type=str, help="Source .pt (e.g. Ultralytics runs/.../best.pt)")
    p.add_argument(
        "--to-models",
        type=str,
        default="",
        help="Destination under models/, e.g. models/yolov8s_traffic.pt",
    )
    p.add_argument(
        "--to-h",
        type=str,
        default="",
        help="Optional dev copy e.g. h/yolov8s.pt",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if checkpoint is not a 5-class traffic head.",
    )
    args = p.parse_args()

    src = Path(args.checkpoint).resolve()
    if not src.is_file():
        print(f"[ERROR] Not a file: {src}", file=sys.stderr)
        return 2

    nc, ordered = _load_names(src)
    names_lower = [x.strip().lower() for x in ordered]
    msg, ok = _verify_task_head(nc, names_lower)
    print(f"[INFO] {msg}\n[INFO] Loaded {src.name}: nc={nc}")

    if args.strict and not ok:
        print("[ERROR] --strict verification failed.", file=sys.stderr)
        return 3

    dests = [Path(x.strip()).resolve() for x in [args.to_models, args.to_h] if x.strip()]
    for dst in dests:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[OK] Copied -> {dst}")

    if not dests:
        print("[WARN] Nothing copied; pass --to-models and/or --to-h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
