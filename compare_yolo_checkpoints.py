from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

EXPECTED_TASK_CLASSES = (
    "motorcycle",
    "person",
    "helmet",
    "no_helmet",
    "license_plate",
)


def default_compare_weight_paths() -> List[str]:
    """Prefer ./actual_weights/best.pt+last.pt, else models/h yolo26 → yolo11 → yolov8."""
    root = Path(__file__).resolve().parent
    candidates: List[str] = []
    for rel in (Path("actual_weights") / "best.pt", Path("actual_weights") / "last.pt"):
        p = (root / rel).resolve()
        if p.is_file():
            candidates.append(str(p))
    if not candidates:
        for folder in ("models", "h"):
            for name in ("yolo26s.pt", "yolo26m.pt", "yolo11s.pt", "yolo11m.pt", "yolov8s.pt", "yolov8m.pt"):
                p = root / folder / name
                if p.is_file():
                    candidates.append(str(p))
    return candidates


def file_size_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024 * 1024))


def summarize_classes(names: Any) -> Tuple[int, List[str]]:
    if isinstance(names, dict):
        ordered = [str(names[i]) for i in sorted(names.keys())]
        return len(ordered), ordered
    if isinstance(names, (list, tuple)):
        lst = [str(x) for x in names]
        return len(lst), lst
    return 0, []


def checkpoint_kind(nc: int, ordered_names: List[str]) -> str:
    if nc == len(EXPECTED_TASK_CLASSES) and tuple(ordered_names) == EXPECTED_TASK_CLASSES:
        return "task_aligned_5cls"
    if nc == 80 and ordered_names[:4] == ["person", "bicycle", "car", "motorcycle"]:
        return "coco_pretrained_80cls"
    return f"custom_or_other_nc_{nc}"


def count_parameters(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def benchmark_latency(
    model: YOLO,
    image: np.ndarray,
    *,
    warmup: int,
    repeats: int,
    conf: float,
    iou: float,
    imgsz: int,
    device: Optional[str],
) -> Dict[str, float]:
    kwargs: Dict[str, Any] = {
        "source": image,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device

    for _ in range(max(0, warmup)):
        model.predict(**kwargs)

    times_ms: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        model.predict(**kwargs)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    p50 = times_ms[n // 2]
    p95 = times_ms[max(0, int(0.95 * n) - 1)]

    return {
        "mean_ms": float(sum(times_ms) / n),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
    }


def build_dummy_image(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def evaluate_one(weights: Path, args: argparse.Namespace) -> Dict[str, Any]:
    model = YOLO(str(weights))
    names_source = getattr(model.model, "names", None) if hasattr(model, "model") else None
    if names_source is None:
        names_source = model.names

    nc, ordered = summarize_classes(names_source)
    kind = checkpoint_kind(nc, ordered)

    image: np.ndarray
    if args.sample_image:
        from importlib import import_module

        cv2 = import_module("cv2")
        image = cv2.imread(str(Path(args.sample_image).resolve()))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.sample_image}")
    else:
        image = build_dummy_image(args.img_h, args.img_w, args.seed)

    latency = benchmark_latency(
        model,
        image,
        warmup=args.warmup,
        repeats=args.repeats,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    out: Dict[str, Any] = {
        "weights": str(weights.resolve()),
        "file_mb": round(file_size_mb(weights), 3),
        "params_m": round(count_parameters(model.model) / 1e6, 3),
        "nc": nc,
        "class_preview": ordered[:8] + (["..."] if len(ordered) > 8 else []),
        "checkpoint_kind": kind,
        "latency_dummy_or_sample_ms": latency,
    }

    if args.data_yaml:
        yaml_path = Path(args.data_yaml).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
        metrics = model.val(data=str(yaml_path), imgsz=args.imgsz, verbose=False, plots=False)
        out["val_metrics"] = {
            "maps": float(getattr(metrics.box, "maps", 0.0) or 0.0),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare YOLO checkpoints: size, params, class head, latency, optional val mAP.",
    )
    default_weights = default_compare_weight_paths()
    if not default_weights:
        default_weights = [str(Path(__file__).resolve().parent / "actual_weights" / "best.pt")]
    p.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=default_weights,
        help="One or more .pt paths (default: actual_weights best/last if present)",
    )
    p.add_argument("--data-yaml", type=str, default="", help="If set, runs model.val() for mAP (needs labels).")
    p.add_argument("--device", type=str, default="", help="e.g. cpu or 0 (empty = Ultralytics default)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--img-h", type=int, default=720)
    p.add_argument("--img-w", type=int, default=1280)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-image", type=str, default="", help="Use real image instead of dummy noise")
    p.add_argument("--json-out", type=str, default="", help="Write machine-readable summary")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_kw = args.device.strip() if args.device else None

    reports: List[Dict[str, Any]] = []
    for w_str in args.weights:
        w = Path(w_str).resolve()
        if not w.exists():
            raise FileNotFoundError(f"Missing weights: {w}")

        snapshot = argparse.Namespace(**{**vars(args), "device": device_kw})
        reports.append(evaluate_one(w, snapshot))

        if reports[-1]["checkpoint_kind"] != "task_aligned_5cls":
            reports[-1]["warning"] = (
                "Checkpoint is not the merged 5 traffic classes. "
                "Use ./actual_weights/best.pt (5-class) or set TRAFFIC_YOLO_WEIGHTS."
            )

    print(json.dumps(reports, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
