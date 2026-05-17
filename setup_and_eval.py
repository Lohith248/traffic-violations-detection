#!/usr/bin/env python3
"""
One entry point: deps → datasets/merged (if missing + API key) → domain mAP JSON.

Windows PowerShell (with download):
  $env:ROBOFLOW_API_KEY='YOUR_KEY'
  python setup_and_eval.py

Already have datasets/merged (e.g. copied from GPU PC):
  python setup_and_eval.py --skip-download
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Deps + dataset (optional) + domain val metrics.")
    p.add_argument("--skip-download", action="store_true", help="Do not call download_models.py")
    p.add_argument("--device", default="", help="Passed through, e.g. 0 or cpu")
    args = p.parse_args()

    py = sys.executable
    merged_yaml = ROOT / "datasets" / "merged" / "data.yaml"
    weights_path = ROOT / "actual_weights" / "best.pt"

    reqs = ROOT / "requirements_download.txt"
    infer = ROOT / "requirements_inference.txt"
    rf: list[str] = []
    if reqs.is_file():
        rf.extend(["-r", str(reqs)])
    if infer.is_file():
        rf.extend(["-r", str(infer)])
    if rf:
        run([py, "-m", "pip", "install", "-q", *rf])

    if not weights_path.is_file():
        print(f"\n[ERROR] Missing weights: {weights_path}", flush=True)
        return 5

    if not merged_yaml.is_file():
        if args.skip_download:
            print("\n[ERROR] No datasets/merged/data.yaml. Ask your friend for the folder or set ROBOFLOW_API_KEY.", flush=True)
            return 3
        key = os.getenv("ROBOFLOW_API_KEY", "").strip()
        if not key or len(key) < 10:
            print(
                "\n[BLOCKED] Need merged dataset.\n"
                "  Option A — set key once, then rerun:\n"
                "    PowerShell:  $env:ROBOFLOW_API_KEY='YOUR_KEY'\n"
                "  Option B — copy SSH machine's entire datasets/merged/ here, then:\n"
                "    python setup_and_eval.py --skip-download\n",
                flush=True,
            )
            return 2
        run([py, str(ROOT / "download_models.py")])

    if not merged_yaml.is_file():
        print(f"\n[ERROR] Still missing: {merged_yaml}", flush=True)
        return 4

    out_json = ROOT / "domain_val_report.json"
    ev = [
        py,
        str(ROOT / "evaluate_merged_dataset.py"),
        "--weights",
        str(weights_path),
        "--data-yaml",
        str(merged_yaml),
        "--json-out",
        str(out_json),
    ]
    if args.device.strip():
        ev.extend(["--device", args.device.strip()])
    run(ev)

    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        snaps = json.dumps(data.get("splits", {}), indent=2)
        print("\n[METRICS SUMMARY]", flush=True)
        print(snaps[:6000], flush=True)
        if len(snaps) > 6000:
            print("… [truncated; see full JSON in file]")
    except Exception:
        print(f"\n[DONE] See full report: {out_json}", flush=True)

    print(f"\n[DONE] {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
