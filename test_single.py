from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from solution import TrafficViolationDetector


def main() -> None:
    try:
        if len(sys.argv) < 2:
            raise ValueError("Usage: python test_single.py <image_path> [model_dir]")

        image_path = Path(sys.argv[1]).resolve()
        model_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else Path("./models").resolve()

        detector = TrafficViolationDetector(model_dir=str(model_dir))

        start = time.perf_counter()
        result = detector.predict(str(image_path))
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        print(json.dumps(result, indent=2))
        print(f"Inference time: {elapsed_ms:.2f} ms")
        if elapsed_ms > 5000.0:
            print("WARNING: Inference exceeded 5000 ms")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
