"""
Download openly reachable photos (Unsplash / Ultralytics mirrors) and run TrafficViolationDetector.

Usage (from repo root):
  python run_inference_cases.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parent
TEST_DIR = PROJECT / "test_cases"

CASES = [
    (
        "01_unsplash_motorcycle_a",
        "Motorcycle road shot (Unsplash); helmets / rider geometry.",
        (
            "https://images.unsplash.com/photo-1568772585407-9361f9bf3a87?auto=format&w=1200&q=80",
        ),
    ),
    (
        "02_unsplash_motorcycle_b",
        "Second motorcycle scene (Unsplash).",
        (
            "https://images.unsplash.com/photo-1583121274602-3e2820c69888?auto=format&w=1200&q=80",
        ),
    ),
    (
        "03_unsplash_landscape_people",
        "Non-traffic baseline (Unsplash landscape with people distant).",
        (
            "https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&w=1200&q=80",
        ),
    ),
    (
        "04_unsplash_car_classic",
        "Classic car (Unsplash); usually no violations (no two-wheelers).",
        (
            "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&w=1200&q=80",
        ),
    ),
    (
        "05_ultralytics_bus",
        "Ultralytics bus.jpg (cars + bus).",
        (
            "https://ultralytics.com/images/bus.jpg",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
        ),
    ),
    (
        "06_ultralytics_zidane",
        "Ultralytics zidane.jpg (soccer players).",
        (
            "https://ultralytics.com/images/zidane.jpg",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg",
        ),
    ),
]


def download_first(urls: tuple[str, ...], dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 TrafficViolationDemo/1.0"
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
                out.write(resp.read())
            return url
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"All URLs failed for {dest.name}: {last_err}") from last_err


def raw_yolo_counts(detector: object, image_path: Path) -> dict:
    """Single forward pass counts by class name (same thresholds as solution.py)."""
    try:
        mod = detector.yolo_model  # type: ignore[attr-defined]
        if mod is None:
            return {}
        image = cv2.imread(str(image_path))
        if image is None:
            return {}
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mod.predict(
            source=rgb,
            conf=0.20,
            iou=0.45,
            imgsz=640,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return {"total_boxes": 0, "per_class": {}}
        boxes = results[0].boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        names = results[0].names
        labels = [str(names[int(i)]) for i in cls_ids]
        per_class = dict(Counter(labels))
        return {"total_boxes": len(labels), "per_class": per_class}
    except Exception as exc:
        return {"error": str(exc)}


def main() -> int:
    sys.path.insert(0, str(PROJECT))
    from solution import TrafficViolationDetector

    TEST_DIR.mkdir(parents=True, exist_ok=True)

    detector = TrafficViolationDetector(model_dir=str(PROJECT / "models"))

    rows = []
    for stem, note, urls in CASES:
        dest = TEST_DIR / f"{stem}.jpg"
        try:
            print(f"[FETCH] {stem}")
            used_url = download_first(urls, dest)
            time.sleep(0.4)
        except Exception as exc:
            rows.append({"case_id": stem, "note": note, "error": str(exc)})
            continue

        t0 = time.perf_counter()
        try:
            result = detector.predict(str(dest))
        except Exception as exc:
            rows.append({"case_id": stem, "note": note, "image": str(dest), "error": str(exc)})
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw = raw_yolo_counts(detector, dest)

        rows.append(
            {
                "case_id": stem,
                "note": note,
                "image": str(dest),
                "source_url": used_url,
                "yolo_checkpoint": detector.yolo_checkpoint_path,
                "ocr_loaded": detector.ocr_model is not None,
                "inference_ms": round(elapsed_ms, 2),
                "violations": result.get("violations", []),
                "violation_count": len(result.get("violations", [])),
                "yolo_detection_summary": raw,
            }
        )

    report_path = PROJECT / "test_cases_inference_report.json"
    report_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"\n[OK] Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

