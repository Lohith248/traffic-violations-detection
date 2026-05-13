from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from roboflow import Roboflow
from ultralytics import YOLO


MODEL_DIR = Path("./models")
DATASET_ROOT = Path("./datasets")
RAW_DATASET_DIR = DATASET_ROOT / "raw"
MERGED_DATASET_DIR = DATASET_ROOT / "merged"

TARGET_CLASSES = ["motorcycle", "person", "helmet", "no_helmet", "license_plate"]

DATASET_IDENTIFIERS = [
    "bike-helmet-detection-2vdjo",
    "helmet-htftb",
    "nckh-2023/helmet-detection-project",
    "ilpd/indian-licence-plate-detection/dataset/4",
    "vehicle-registration-plates-trudk",
]

PADDLE_ASSETS = {
    "paddle_det": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
    "paddle_rec": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
    "paddle_cls": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
}

CLASS_ALIASES = {
    "motorcycle": {"motorcycle", "bike", "scooter", "motorbike", "two_wheeler", "two-wheeler"},
    "person": {"person", "rider", "human", "people", "man", "woman"},
    "helmet": {"helmet", "with_helmet", "helmet_on", "wearing_helmet"},
    "no_helmet": {"no_helmet", "without_helmet", "nohelmet", "not_wearing_helmet", "helmet_off"},
    "license_plate": {
        "license_plate",
        "licence_plate",
        "plate",
        "number_plate",
        "registration_plate",
        "numberplate",
        "licenceplate",
        "licenseplate",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def file_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def format_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def print_model_sizes(model_dir: Path) -> None:
    print("\n[INFO] Model size report")
    tracked = [
        model_dir / "yolov8s.pt",
        model_dir / "paddle_det",
        model_dir / "paddle_rec",
        model_dir / "paddle_cls",
    ]
    for item in tracked:
        print(f" - {item}: {format_mb(file_size_bytes(item))}")

    total = file_size_bytes(model_dir)
    limit = 250 * 1024 * 1024
    status = "OK" if total <= limit else "EXCEEDS LIMIT"
    print(f" - TOTAL {model_dir}: {format_mb(total)} ({status}, limit = 250.00 MB)\n")


def download_url(url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def find_infer_dir(root: Path) -> Optional[Path]:
    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        if (candidate / "inference.pdmodel").exists() and (candidate / "inference.pdiparams").exists():
            return candidate
    return None


def extract_tar_to_flat_dir(archive_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(tmp_dir)

        infer_dir = find_infer_dir(tmp_dir)
        if infer_dir is None:
            raise RuntimeError(f"Could not find Paddle inference files in {archive_path}")

        for item in output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)

        for src in infer_dir.iterdir():
            dst = output_dir / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)


def download_yolov8_base_weights(model_dir: Path) -> Path:
    ensure_dir(model_dir)
    output = model_dir / "yolov8s.pt"
    if output.exists():
        print(f"[SKIP] {output} already exists")
        return output

    print("[INFO] Downloading YOLOv8s base weights...")
    model = YOLO("yolov8s.pt")
    cache_path = Path(str(model.ckpt_path))
    if not cache_path.exists():
        raise FileNotFoundError("Ultralytics did not provide a valid checkpoint path.")
    shutil.copy2(cache_path, output)
    print(f"[OK] Saved {output}")
    return output


def download_paddle_models(model_dir: Path) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    ensure_dir(model_dir)

    for folder_name, url in PADDLE_ASSETS.items():
        target_dir = model_dir / folder_name
        outputs[folder_name] = target_dir
        if (target_dir / "inference.pdmodel").exists() and (target_dir / "inference.pdiparams").exists():
            print(f"[SKIP] {target_dir} already exists")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / f"{folder_name}.tar"
            print(f"[INFO] Downloading {folder_name} ...")
            download_url(url, archive_path)
            print(f"[INFO] Extracting {folder_name} -> {target_dir} ...")
            extract_tar_to_flat_dir(archive_path, target_dir)

        print(f"[OK] Saved {target_dir}")

    return outputs


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def map_source_class_to_target_id(name: str) -> Optional[int]:
    normalized = normalize_name(name)
    for idx, target_name in enumerate(TARGET_CLASSES):
        if normalized in CLASS_ALIASES[target_name]:
            return idx
    return None


def parse_names_from_data_yaml(data_yaml_path: Path) -> Dict[int, str]:
    content = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    names = content.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(val) for idx, val in enumerate(names)}
    return {}


def remap_label_file(source_label: Path, destination_label: Path, source_names: Dict[int, str]) -> None:
    ensure_dir(destination_label.parent)
    if not source_label.exists():
        destination_label.write_text("", encoding="utf-8")
        return

    out_lines: List[str] = []
    for raw in source_label.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            src_cls = int(float(parts[0]))
        except ValueError:
            continue
        mapped = map_source_class_to_target_id(source_names.get(src_cls, ""))
        if mapped is None:
            continue
        out_lines.append(f"{mapped} {' '.join(parts[1:])}")

    destination_label.write_text("\n".join(out_lines), encoding="utf-8")


def split_candidates() -> Dict[str, List[str]]:
    return {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}


def merge_datasets(downloaded_roots: List[Path], merged_root: Path) -> Path:
    for split in ["train", "val", "test"]:
        ensure_dir(merged_root / split / "images")
        ensure_dir(merged_root / split / "labels")

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates = split_candidates()

    for ds_idx, dataset_root in enumerate(downloaded_roots):
        data_yaml = dataset_root / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Missing data.yaml in dataset: {dataset_root}")
        source_names = parse_names_from_data_yaml(data_yaml)

        for target_split, src_split_candidates in candidates.items():
            source_split = next((s for s in src_split_candidates if (dataset_root / s / "images").exists()), None)
            if source_split is None:
                continue

            src_images = dataset_root / source_split / "images"
            src_labels = dataset_root / source_split / "labels"

            for image_path in src_images.iterdir():
                if image_path.suffix.lower() not in image_suffixes:
                    continue

                prefix = f"ds{ds_idx}_{dataset_root.name}_{image_path.stem}"
                dest_image = merged_root / target_split / "images" / f"{prefix}{image_path.suffix.lower()}"
                dest_label = merged_root / target_split / "labels" / f"{prefix}.txt"
                src_label = src_labels / f"{image_path.stem}.txt"

                shutil.copy2(image_path, dest_image)
                remap_label_file(src_label, dest_label, source_names)

    merged_yaml = merged_root / "data.yaml"
    merged_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(merged_root.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": TARGET_CLASSES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return merged_yaml


def api_get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def try_extract_workspace_project(payload: Any) -> Optional[Tuple[str, str]]:
    if isinstance(payload, dict):
        project = payload.get("project")
        if isinstance(project, dict):
            project_id = project.get("id")
            if isinstance(project_id, str):
                parts = project_id.split("/")
                if len(parts) >= 2:
                    return parts[0], parts[1]

        for value in payload.values():
            result = try_extract_workspace_project(value)
            if result is not None:
                return result

    if isinstance(payload, list):
        for item in payload:
            result = try_extract_workspace_project(item)
            if result is not None:
                return result

    if isinstance(payload, str):
        if payload.startswith("http"):
            return None
        parts = payload.split("/")
        if len(parts) >= 2 and all(parts[:2]):
            workspace, project = parts[0], parts[1]
            if re.match(r"^[a-zA-Z0-9\-_]+$", workspace) and re.match(r"^[a-zA-Z0-9\-_]+$", project):
                return workspace, project
    return None


def get_accessible_workspaces(api_key: str) -> List[str]:
    workspaces: List[str] = []
    try:
        payload = api_get_json(f"https://api.roboflow.com/?api_key={urllib.parse.quote(api_key)}")
        if isinstance(payload, dict):
            ws = payload.get("workspace")
            if isinstance(ws, str):
                workspaces.append(ws)
            maybe_workspaces = payload.get("workspaces")
            if isinstance(maybe_workspaces, dict):
                for _, ws_info in maybe_workspaces.items():
                    if isinstance(ws_info, dict):
                        url = ws_info.get("url")
                        if isinstance(url, str):
                            workspaces.append(url)
    except Exception:
        pass
    return sorted(set(workspaces))


def resolve_project_from_slug(slug: str, api_key: str, workspace_hints: List[str]) -> Tuple[str, str]:
    encoded_key = urllib.parse.quote(api_key)
    encoded_slug = urllib.parse.quote(slug)

    api_candidates = [
        f"https://api.roboflow.com/{encoded_slug}?api_key={encoded_key}",
        f"https://api.roboflow.com/dataset/{encoded_slug}?api_key={encoded_key}",
    ]
    for url in api_candidates:
        try:
            payload = api_get_json(url)
            pair = try_extract_workspace_project(payload)
            if pair is not None:
                return pair
        except Exception:
            continue

    for workspace in workspace_hints:
        try:
            payload = api_get_json(
                f"https://api.roboflow.com/{urllib.parse.quote(workspace)}/{encoded_slug}?api_key={encoded_key}"
            )
            pair = try_extract_workspace_project(payload)
            if pair is not None:
                return pair
        except Exception:
            continue

    raise RuntimeError(
        f"Could not resolve workspace/project for '{slug}'. "
        f"Use an API key that can access this dataset or provide the full workspace/project identifier."
    )


def parse_dataset_identifier(
    dataset_id: str, api_key: str, workspace_hints: List[str]
) -> Tuple[str, str, Optional[int]]:
    match_full = re.match(r"^([^/]+)/([^/]+)/dataset/(\d+)$", dataset_id)
    if match_full:
        return match_full.group(1), match_full.group(2), int(match_full.group(3))

    match_ws_proj = re.match(r"^([^/]+)/([^/]+)$", dataset_id)
    if match_ws_proj:
        return match_ws_proj.group(1), match_ws_proj.group(2), None

    workspace, project = resolve_project_from_slug(dataset_id, api_key, workspace_hints)
    return workspace, project, None


def get_latest_version_number(project_obj: Any) -> int:
    infos = project_obj.get_version_information()
    version_numbers: List[int] = []
    for item in infos:
        version_id = str(item.get("id", ""))
        tail = version_id.split("/")[-1]
        if tail.isdigit():
            version_numbers.append(int(tail))
    if not version_numbers:
        raise RuntimeError(f"No downloadable versions found for project: {project_obj.id}")
    return max(version_numbers)


def download_roboflow_datasets(api_key: str, output_raw_dir: Path) -> List[Path]:
    rf = Roboflow(api_key=api_key)
    ensure_dir(output_raw_dir)
    workspace_hints = get_accessible_workspaces(api_key)
    downloaded_roots: List[Path] = []

    for dataset_id in DATASET_IDENTIFIERS:
        workspace, project, explicit_version = parse_dataset_identifier(dataset_id, api_key, workspace_hints)
        project_obj = rf.workspace(workspace).project(project)
        version = explicit_version if explicit_version is not None else get_latest_version_number(project_obj)
        version_obj = project_obj.version(version)

        local_name = f"{workspace}__{project}__v{version}"
        local_dir = output_raw_dir / local_name
        print(f"[INFO] Downloading Roboflow dataset: {dataset_id} -> {workspace}/{project} (v{version})")

        dataset = version_obj.download("yolov8", location=str(local_dir), overwrite=False)
        downloaded_roots.append(Path(dataset.location))
        print(f"[OK] Saved dataset to {dataset.location}")

    return downloaded_roots


def main() -> None:
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Environment variable ROBOFLOW_API_KEY is required.")

    print("[STEP] Downloading YOLOv8s base weights...")
    download_yolov8_base_weights(MODEL_DIR)

    print("[STEP] Downloading PaddleOCR det/rec/cls models...")
    download_paddle_models(MODEL_DIR)

    print("[STEP] Downloading Roboflow datasets...")
    roots = download_roboflow_datasets(api_key=api_key, output_raw_dir=RAW_DATASET_DIR)

    print("[STEP] Merging datasets into 5 target classes...")
    merged_yaml = merge_datasets(roots, MERGED_DATASET_DIR)
    print(f"[OK] Merged dataset YAML: {merged_yaml}")

    print_model_sizes(MODEL_DIR)
    print("[DONE] Offline setup is complete.")


if __name__ == "__main__":
    main()
