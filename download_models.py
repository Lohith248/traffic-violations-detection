from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml
from ultralytics import YOLO

try:
    from roboflow import Roboflow
except ImportError:
    Roboflow = None  # type: ignore[assignment]


MODEL_DIR = Path("./models")
DATASET_ROOT = Path("./datasets")
RAW_DATASET_DIR = DATASET_ROOT / "raw"
MERGED_DATASET_DIR = DATASET_ROOT / "merged"

TARGET_CLASSES = ["motorcycle", "person", "helmet", "no_helmet", "license_plate"]

# Roboflow workspace/project slug or slug-only (resolved via API). Order: multi-class helmets/riders first,
# then plates. Slugs drift on Universe — if download fails, check the project's URL and paste the exact path.
#
# Curated selection rationale:
#  1. abhilash-poojary  — ONLY dataset with all 5 target classes in a single bundle
#  2. traffic-violation/with-no-helmet — strong motorcycle + no_helmet + plate coverage (~350 imgs)
#  3. traffic-violation/no-helmet-jkn77 — pure no_helmet boost (~144 imgs)
#  4. triplerider — motorcycle + helmet/no_helmet in triple-riding scenarios (~582 imgs)
#  5. nckh-2023/helmet-detection-project — already downloaded, good helmet + person + plate (~1563 imgs)
#  6. vehicle-registration-plates-trudk — massive clean plate dataset (~8800 imgs)
#
# DROPPED:
#  - ilpd/indian-licence-plate-detection: character-level OCR annotations (A-Z, 0-9), NOT plate bboxes
#  - bike-helmet-detection-2vdjo: redundant with nckh-2023, download failures
#  - expertos/helmet-motorcycle-no-helmet-person: noisy "objects" catch-all class
DATASET_IDENTIFIERS = [
    "abhilash-poojary/person-dataset-5qprs",
    "traffic-violation/with-no-helmet",
    "traffic-violation/no-helmet-jkn77",
    "triplerider/triplerider",
    "nckh-2023/helmet-detection-project",
    "vehicle-registration-plates-trudk",
]

PADDLE_ASSETS = {
    "paddle_det": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
    "paddle_rec": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
    "paddle_cls": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
}

MAX_BACKGROUND_RATIO_PER_SPLIT = 0.35

CLASS_ALIASES = {
    "motorcycle": {
        "motorcycle",
        "motor_cycle",
        "bike",
        "scooter",
        "motorbike",
        "moped",
        "two_wheeler",
        "two-wheeler",
        "2_wheeler",
        "2wheeler",
        # Traffic/violation dataset spellings (names run through normalize_name → lowercase + underscores).
        "motor_cycles",
        "two_wheelers",
        # Kaggle dataset variations
        "motorcycle_with_rider",
        "vehicle",  # hozngvan dataset sometimes labels motorcycle as "vehicle"
    },
    "person": {
        "person",
        "rider",
        "human",
        "people",
        "man",
        "woman",
        "driver",
        "passenger",
        "pillion",
        "pillion_rider",
        # Rider-as-one-box datasets (Roboflow helmet projects often label this vs "motorcycle").
        "motorcyclist",
        "motor_cyclist",
        "motorcycle_rider",
        "bike_rider",
        "motor_rider",
        "biker",
        # Triple riding / multiple riders (same head as "people on bike" unless you split triple later).
        "tripling",
        "triple_riding",
        "tripleriding",
        "triple_riders",
        "triple_rider",
        # devgurucodes / meliodassourav Kaggle datasets
        "overloading",  # overloaded bike = multiple persons on motorcycle
        "overload",
    },
    "helmet": {
        "helmet",
        "with_helmet",
        "withhelmet",
        "helmet_on",
        "wearing_helmet",
        # CamelCase variants become these after normalization.
        "helmets",
        # Kaggle dataset variations
        "with helmet",
        "with_helment",  # common typos
    },
    "no_helmet": {
        "no_helmet",
        "without_helmet",
        "withouthelmet",
        "nohelmet",
        "not_wearing_helmet",
        "helmet_off",
        "unhelmeted",
        "bare_head",
        "barehead",
        "no_helm",
        "nohelm",
        "no_helment",  # common typo on small datasets
        # Kaggle dataset variations
        "without helmet",
        "without_helment",  # common typo
        "head",  # some datasets label bare head as "head"
    },
    "license_plate": {
        "license_plate",
        "licence_plate",
        "license",
        "licence",
        "plate",
        "number_plate",
        "registration_plate",
        "registration_number",
        "numberplate",
        "licenceplate",
        "licenseplate",
        # Traffic Violation / Indian plate naming
        "platenumber",
        "plate_number",
        "plateno",
        "plate_no",
        "lp",
        # Kaggle dataset variations
        "number plate",
        "indian_licence_plate",
        "indian_license_plate",
        "reg_plate",
    },
}


def verify_roboflow_api_key(api_key: str) -> None:
    """Verifies that the API key is valid and not a placeholder."""
    if not api_key or len(api_key) < 10 or api_key.startswith("YOUR_"):
        raise ValueError(
            f"Invalid ROBOFLOW_API_KEY: '{api_key}'. "
            "Please export a valid key from your Roboflow dashboard."
        )

    try:
        url = f"https://api.roboflow.com/?api_key={urllib.parse.quote(api_key)}"
        api_get_json(url)
    except Exception as e:
        raise RuntimeError(f"Could not verify Roboflow API key. Check your internet and key: {e}")


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
        model_dir / "yolo26s.pt",
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


DEFAULT_YOLO_BASE_WEIGHT = "yolo26s.pt"


def download_default_yolo_base_weights(model_dir: Path, hub_name: str = DEFAULT_YOLO_BASE_WEIGHT) -> Path:
    ensure_dir(model_dir)
    output = model_dir / hub_name
    if output.exists():
        print(f"[SKIP] {output} already exists")
        return output

    print(f"[INFO] Downloading YOLO base weights ({hub_name})...")
    model = YOLO(hub_name)
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
    # Explicit drops (datasets include catch-all buckets we should not coerce into targets).
    if normalized in {"objects", "object", "misc", "other"}:
        return None
    for idx, target_name in enumerate(TARGET_CLASSES):
        if normalized in CLASS_ALIASES[target_name]:
            return idx

    # Partial / compound names (underscore tokenization did not isolate the word).
    if "motorcyclist" in normalized:
        return TARGET_CLASSES.index("person")

    tokens = [t for t in normalized.split("_") if t]
    token_set = set(tokens)

    if any(t in normalized for t in ["plate", "licence", "license", "registration"]):
        return TARGET_CLASSES.index("license_plate")

    if "helmet" in normalized:
        negative_tokens = {"no", "without", "non", "off", "unhelmeted", "bare"}
        if token_set.intersection(negative_tokens) or "nohelmet" in normalized or "withouthelmet" in normalized:
            return TARGET_CLASSES.index("no_helmet")
        return TARGET_CLASSES.index("helmet")

    if token_set.intersection({"person", "rider", "human", "man", "woman", "driver", "passenger", "pillion"}):
        return TARGET_CLASSES.index("person")

    if token_set.intersection({"motorcycle", "motor", "bike", "motorbike", "scooter", "two", "wheeler"}):
        return TARGET_CLASSES.index("motorcycle")

    return None


def parse_names_from_data_yaml(data_yaml_path: Path) -> Dict[int, str]:
    content = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    names = content.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(val) for idx, val in enumerate(names)}
    return {}


def sanitize_yolo_bbox(xc: float, yc: float, bw: float, bh: float) -> Optional[Tuple[float, float, float, float]]:
    eps = 1e-6
    if bw <= 0.0 or bh <= 0.0:
        return None

    x1 = xc - (bw / 2.0)
    y1 = yc - (bh / 2.0)
    x2 = xc + (bw / 2.0)
    y2 = yc + (bh / 2.0)

    x1 = min(1.0 - eps, max(0.0, x1))
    y1 = min(1.0 - eps, max(0.0, y1))
    x2 = min(1.0, max(x1 + eps, x2))
    y2 = min(1.0, max(y1 + eps, y2))

    return (
        min(1.0, max(0.0, (x1 + x2) / 2.0)),
        min(1.0, max(0.0, (y1 + y2) / 2.0)),
        min(1.0, max(eps, x2 - x1)),
        min(1.0, max(eps, y2 - y1)),
    )


def remap_label_file(source_label: Path, destination_label: Path, source_names: Dict[int, str]) -> List[int]:
    ensure_dir(destination_label.parent)
    if not source_label.exists():
        destination_label.write_text("", encoding="utf-8")
        return []

    out_lines: List[str] = []
    mapped_classes: List[int] = []
    for raw in source_label.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            src_cls = int(float(parts[0]))
            xc, yc, bw, bh = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        mapped = map_source_class_to_target_id(source_names.get(src_cls, ""))
        if mapped is None:
            continue

        sanitized = sanitize_yolo_bbox(xc, yc, bw, bh)
        if sanitized is None:
            continue

        sxc, syc, sbw, sbh = sanitized
        out_lines.append(f"{mapped} {sxc:.6f} {syc:.6f} {sbw:.6f} {sbh:.6f}")
        mapped_classes.append(mapped)

    destination_label.write_text("\n".join(out_lines), encoding="utf-8")
    return mapped_classes


def split_candidates() -> Dict[str, List[str]]:
    return {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}


def collect_split_statistics(merged_root: Path, split: str) -> Dict[str, Any]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images_dir = merged_root / split / "images"
    labels_dir = merged_root / split / "labels"
    class_counts = [0 for _ in TARGET_CLASSES]
    labeled_images = 0
    background_images = 0

    if not images_dir.exists():
        return {
            "images": 0,
            "labeled_images": 0,
            "background_images": 0,
            "class_counts": class_counts,
            "boxes": 0,
        }

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in image_suffixes:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        valid_line_count = 0
        if label_path.exists():
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= cls_id < len(class_counts):
                    class_counts[cls_id] += 1
                    valid_line_count += 1
        if valid_line_count > 0:
            labeled_images += 1
        else:
            background_images += 1

    return {
        "images": labeled_images + background_images,
        "labeled_images": labeled_images,
        "background_images": background_images,
        "class_counts": class_counts,
        "boxes": sum(class_counts),
    }


def trim_background_images(merged_root: Path, split: str, max_background_ratio: float) -> None:
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images_dir = merged_root / split / "images"
    labels_dir = merged_root / split / "labels"
    if not images_dir.exists():
        return

    background_images: List[Path] = []
    positive_count = 0
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in image_suffixes:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        is_positive = False
        if label_path.exists():
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw.strip().split()
                if len(parts) == 5:
                    is_positive = True
                    break
        if is_positive:
            positive_count += 1
        else:
            background_images.append(image_path)

    allowed_background = int(round(positive_count * max_background_ratio))
    excess = max(0, len(background_images) - allowed_background)
    if excess == 0:
        return

    for image_path in background_images[-excess:]:
        label_path = labels_dir / f"{image_path.stem}.txt"
        image_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)

    print(
        f"[INFO] Trimmed {excess} background-only images from split '{split}' "
        f"(kept {allowed_background}, positive={positive_count})."
    )


def print_dataset_statistics(merged_root: Path) -> Dict[str, Dict[str, Any]]:
    all_stats: Dict[str, Dict[str, Any]] = {}
    print("\n[INFO] Merged dataset statistics")
    for split in ["train", "val", "test"]:
        stats = collect_split_statistics(merged_root, split)
        all_stats[split] = stats
        class_repr = {TARGET_CLASSES[i]: stats["class_counts"][i] for i in range(len(TARGET_CLASSES))}
        print(
            f" - {split}: images={stats['images']}, labeled={stats['labeled_images']}, "
            f"background={stats['background_images']}, boxes={stats['boxes']}, classes={class_repr}"
        )
    print("")
    return all_stats


def validate_split_coverage(all_stats: Dict[str, Dict[str, Any]], *, strict_train: bool = True) -> None:
    train_counts = all_stats["train"]["class_counts"]
    missing_train = [TARGET_CLASSES[i] for i, c in enumerate(train_counts) if c <= 0]
    if missing_train:
        msg = (
            "Merged train split is missing target classes: "
            f"{missing_train}. Fix class mapping/datasets or pass --allow-incomplete-merge "
            "(not recommended for final training)."
        )
        if strict_train:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")

    for split in ["val", "test"]:
        counts = all_stats[split]["class_counts"]
        missing = [TARGET_CLASSES[i] for i, c in enumerate(counts) if c <= 0]
        if missing:
            print(f"[WARN] Split '{split}' is missing classes: {missing}")


def merge_datasets(downloaded_roots: List[Path], merged_root: Path, *, strict_train: bool = True) -> Path:
    if merged_root.exists():
        shutil.rmtree(merged_root, ignore_errors=True)

    try:
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

        for split in ["train", "val", "test"]:
            trim_background_images(merged_root, split, max_background_ratio=MAX_BACKGROUND_RATIO_PER_SPLIT)

        stats = print_dataset_statistics(merged_root)
        validate_split_coverage(stats, strict_train=strict_train)

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
    except BaseException:
        shutil.rmtree(merged_root, ignore_errors=True)
        raise



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


def _attempt_download(version_obj: Any, local_dir: Path) -> Optional[Path]:
    """
    Download a Roboflow dataset version as YOLO (v8 then v5 fallback).
    Many projects lack a working yolov8 cloud export or pin an old ultralytics;
    yolov5 exports still unpack to the same layout for merge.
    """
    last_exc: Optional[Exception] = None

    for fmt in ("yolov8", "yolov5"):
        shutil.rmtree(local_dir, ignore_errors=True)

        try:
            dataset = version_obj.download(fmt, location=str(local_dir), overwrite=True)
            result_path = Path(dataset.location)
            if (result_path / "data.yaml").exists():
                print(f"[INFO] Downloaded as Roboflow format={fmt!r} -> {result_path}")
                return result_path
            shutil.rmtree(local_dir, ignore_errors=True)
            print(f"[WARN] {fmt} download completed but data.yaml is missing; trying next format...")
        except Exception as e:
            last_exc = e
            no_such_key = False
            zip_path = local_dir / "roboflow.zip"
            if zip_path.exists():
                try:
                    with zip_path.open("rb") as f:
                        chunk = f.read(500)
                    decoded = chunk.decode("utf-8", errors="ignore")
                    if "NoSuchKey" in decoded or "does not exist" in decoded.lower():
                        no_such_key = True
                except Exception:
                    pass

            shutil.rmtree(local_dir, ignore_errors=True)

            if fmt == "yolov8":
                detail = "export missing" if no_such_key else str(e).split("\n")[0][:120]
                print(f"[WARN] yolov8 download failed ({detail}); trying yolov5...")
                continue

            if no_such_key:
                return None
            if last_exc is not None:
                raise last_exc

    return None


def download_roboflow_datasets(
    api_key: str,
    output_raw_dir: Path,
    dataset_identifiers: Optional[List[str]] = None,
) -> List[Path]:
    if Roboflow is None:
        raise ImportError(
            "roboflow package is required to download datasets. "
            "Install with: pip install -r requirements_download.txt"
        )

    rf = Roboflow(api_key=api_key)
    ensure_dir(output_raw_dir)
    workspace_hints = get_accessible_workspaces(api_key)
    downloaded_roots: List[Path] = []
    skipped: List[str] = []

    ids = dataset_identifiers if dataset_identifiers is not None else list(DATASET_IDENTIFIERS)
    for dataset_id in ids:
        workspace, project, explicit_version = parse_dataset_identifier(dataset_id, api_key, workspace_hints)
        project_obj = rf.workspace(workspace).project(project)

        # Build the list of versions to try
        if explicit_version is not None:
            versions_to_try = [explicit_version]
        else:
            all_versions = sorted(
                [
                    int(str(item.get("id", "")).split("/")[-1])
                    for item in project_obj.get_version_information()
                    if str(item.get("id", "")).split("/")[-1].isdigit()
                ],
                reverse=True,  # newest first
            )
            versions_to_try = all_versions if all_versions else []

        if not versions_to_try:
            print(f"[WARN] No versions found for {dataset_id} — skipping.")
            skipped.append(dataset_id)
            continue

        dataset_root = None
        for version in versions_to_try:
            local_name = f"{workspace}__{project}__v{version}"
            local_dir = output_raw_dir / local_name

            # Already successfully downloaded
            if (local_dir / "data.yaml").exists():
                print(f"[SKIP] Dataset already exists at {local_dir}")
                dataset_root = local_dir
                break

            print(f"[INFO] Downloading Roboflow dataset: {dataset_id} -> {workspace}/{project} (v{version})")
            result = _attempt_download(project_obj.version(version), local_dir)
            if result is not None:
                print(f"[OK] Saved dataset to {result}")
                dataset_root = result
                break
            else:
                print(f"[WARN] v{version} of {dataset_id} has no usable YOLO export — trying older version...")

        if dataset_root is not None:
            downloaded_roots.append(dataset_root)
        else:
            print(f"[WARN] Could not download any version of {dataset_id} — skipping.")
            skipped.append(dataset_id)

    if skipped:
        print(f"\n[WARN] {len(skipped)} dataset(s) were skipped: {skipped}")

    if not downloaded_roots:
        raise RuntimeError("No datasets were downloaded successfully. Cannot proceed with merging.")

    return downloaded_roots


def discover_existing_raw_roots(raw_dir: Path) -> List[Path]:
    """Subfolders under raw_dir that contain a YOLO data.yaml."""
    if not raw_dir.is_dir():
        return []
    return sorted(p for p in raw_dir.iterdir() if p.is_dir() and (p / "data.yaml").is_file())


def merged_dataset_identifier_list(cli_extra: str = "") -> List[str]:
    """DATASET_IDENTIFIERS plus optional comma-separated slug list (CLI/env)."""
    ids = list(DATASET_IDENTIFIERS)
    extra = (cli_extra or "").strip()
    if not extra:
        extra = os.getenv("ROBOFLOW_EXTRA_DATASETS", "").strip()
    if extra:
        ids.extend(part.strip() for part in extra.split(",") if part.strip())
    return ids


def parse_dm_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Paddle + YOLO base weights + Roboflow dumps, merge into datasets/merged."
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip downloads; rebuild datasets/merged from existing datasets/raw/*/data.yaml only",
    )
    p.add_argument(
        "--allow-incomplete-merge",
        action="store_true",
        help="Warn instead of failing when merged train lacks any of the 5 target classes",
    )
    p.add_argument(
        "--extra-datasets",
        type=str,
        default="",
        help=(
            "Comma-separated Roboflow dataset slugs; appended after built-in DATASET_IDENTIFIERS. "
            "When empty, ROBOFLOW_EXTRA_DATASETS env is used if set."
        ),
    )
    p.add_argument(
        "--no-paddle",
        action="store_true",
        help="Skip PaddleOCR model download (datasets still merged if downloaded)",
    )
    p.add_argument(
        "--no-yolov8-base",
        action="store_true",
        help="Skip copying default yolo26s.pt into models/",
    )
    return p.parse_args(argv)


def main() -> None:
    args = parse_dm_args()
    strict_train = not args.allow_incomplete_merge

    if args.merge_only:
        roots = discover_existing_raw_roots(RAW_DATASET_DIR)
        if not roots:
            raise RuntimeError(
                f"No YOLO dataset roots found under {RAW_DATASET_DIR} (each subfolder needs data.yaml). "
                "Run a full download first or copy Roboflow exports there."
            )
        print(f"[STEP] Merge-only: rebuilding {MERGED_DATASET_DIR} from {len(roots)} raw dataset(s)...")
        merge_datasets(roots, MERGED_DATASET_DIR, strict_train=strict_train)
        print(f"[OK] Merged YAML: {MERGED_DATASET_DIR / 'data.yaml'}")
        return

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    verify_roboflow_api_key(api_key)

    if not args.no_yolov8_base:
        print("[STEP] Downloading default YOLO26s base weights into models/ …")
        download_default_yolo_base_weights(MODEL_DIR)

    if not args.no_paddle:
        print("[STEP] Downloading PaddleOCR det/rec/cls models...")
        download_paddle_models(MODEL_DIR)

    identifiers = merged_dataset_identifier_list(args.extra_datasets)

    print("[STEP] Downloading Roboflow datasets...")
    roots = download_roboflow_datasets(api_key=api_key, output_raw_dir=RAW_DATASET_DIR, dataset_identifiers=identifiers)

    # Also include any Kaggle-converted datasets (from convert_kaggle_to_yolo.py)
    kaggle_roots = [
        p for p in discover_existing_raw_roots(RAW_DATASET_DIR)
        if p not in roots and "kaggle" in p.name.lower()
    ]
    if kaggle_roots:
        print(f"[INFO] Also including {len(kaggle_roots)} Kaggle dataset(s): {[r.name for r in kaggle_roots]}")
        roots.extend(kaggle_roots)

    print("[STEP] Merging datasets into 5 target classes...")
    merged_yaml = merge_datasets(roots, MERGED_DATASET_DIR, strict_train=strict_train)
    print(f"[OK] Merged dataset YAML: {merged_yaml}")

    print_model_sizes(MODEL_DIR)
    print("[DONE] Offline setup is complete.")


if __name__ == "__main__":
    main()
