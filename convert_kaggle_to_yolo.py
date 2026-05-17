#!/usr/bin/env python3
"""
Convert Kaggle datasets to YOLO-compatible layout for merging.

Supports K1-K8 Kaggle datasets. Auto-detects VOC XML or YOLO format.
After running, do: python download_models.py --merge-only
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

KAGGLE_RAW_BASE = Path("./datasets/raw")
TARGET_CLASSES = ["motorcycle", "person", "helmet", "no_helmet", "license_plate"]

CLASS_MAP = {
    "motorcycle": "motorcycle", "bike": "motorcycle", "motorbike": "motorcycle",
    "two_wheeler": "motorcycle", "two-wheeler": "motorcycle", "scooter": "motorcycle",
    "motor_cycle": "motorcycle", "motorcycle_with_rider": "motorcycle", "vehicle": "motorcycle",
    "person": "person", "rider": "person", "human": "person", "motorcyclist": "person",
    "biker": "person", "driver": "person", "pillion": "person",
    "tripling": "person", "triple_riding": "person", "tripleriding": "person",
    "overloading": "person", "overload": "person",
    "helmet": "helmet", "with_helmet": "helmet", "withhelmet": "helmet",
    "with helmet": "helmet", "With Helmet": "helmet", "WithHelmet": "helmet",
    "wearing_helmet": "helmet", "helmets": "helmet",
    "no_helmet": "no_helmet", "without_helmet": "no_helmet", "withouthelmet": "no_helmet",
    "without helmet": "no_helmet", "Without Helmet": "no_helmet", "WithoutHelmet": "no_helmet",
    "nohelmet": "no_helmet", "no-helmet": "no_helmet", "bare_head": "no_helmet",
    "head": "no_helmet",
    "license_plate": "license_plate", "licence_plate": "license_plate",
    "number_plate": "license_plate", "number plate": "license_plate",
    "Number Plate": "license_plate", "plate": "license_plate", "Plate": "license_plate",
    "numberplate": "license_plate", "LP": "license_plate", "lp": "license_plate",
    "indian_licence_plate": "license_plate", "indian_license_plate": "license_plate",
    "reg_plate": "license_plate",
}


def normalize_class_name(name: str) -> Optional[str]:
    if name in CLASS_MAP:
        return CLASS_MAP[name]
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in CLASS_MAP:
        return CLASS_MAP[normalized]
    if "helmet" in normalized:
        negatives = {"no", "without", "non", "off", "bare"}
        if set(normalized.split("_")) & negatives:
            return "no_helmet"
        return "helmet"
    if any(k in normalized for k in ["plate", "licence", "license", "registration"]):
        return "license_plate"
    if any(k in normalized for k in ["rider", "person", "human", "overload", "triple"]):
        return "person"
    if any(k in normalized for k in ["motorcycle", "bike", "scooter", "motorbike"]):
        return "motorcycle"
    return None


def target_class_id(target_name: str) -> int:
    return TARGET_CLASSES.index(target_name)


def convert_voc_to_yolo(xml_path: Path, output_label_path: Path) -> bool:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            return False
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
        if width <= 0 or height <= 0:
            return False
        lines: List[str] = []
        for obj in root.iter("object"):
            class_name_raw = obj.findtext("name", "").strip()
            target = normalize_class_name(class_name_raw)
            if target is None:
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
            x_center = max(0.0, min(1.0, (xmin + xmax) / 2.0 / width))
            y_center = max(0.0, min(1.0, (ymin + ymax) / 2.0 / height))
            bbox_w = max(0.001, min(1.0, (xmax - xmin) / width))
            bbox_h = max(0.001, min(1.0, (ymax - ymin) / height))
            cls_id = target_class_id(target)
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")
        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        output_label_path.write_text("\n".join(lines), encoding="utf-8")
        return len(lines) > 0
    except Exception as e:
        print(f"[WARN] Failed to parse {xml_path}: {e}")
        return False


def convert_yolo_labels_with_remap(
    source_label: Path, dest_label: Path, source_class_names: Dict[int, str],
) -> bool:
    if not source_label.exists():
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        dest_label.write_text("", encoding="utf-8")
        return False
    lines: List[str] = []
    for raw in source_label.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            src_cls = int(float(parts[0]))
            xc, yc, bw, bh = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        class_name = source_class_names.get(src_cls, "")
        target = normalize_class_name(class_name)
        if target is None:
            continue
        cls_id = target_class_id(target)
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    dest_label.parent.mkdir(parents=True, exist_ok=True)
    dest_label.write_text("\n".join(lines), encoding="utf-8")
    return len(lines) > 0


def discover_images(directory: Path) -> List[Path]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for suffix in image_suffixes:
        images.extend(directory.rglob(f"*{suffix}"))
        images.extend(directory.rglob(f"*{suffix.upper()}"))
    return sorted(set(images))


def split_images(images: List[Path], train_ratio: float = 0.8, val_ratio: float = 0.15, seed: int = 42):
    random.seed(seed)
    shuffled = list(images)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {"train": shuffled[:n_train], "val": shuffled[n_train:n_train + n_val], "test": shuffled[n_train + n_val:]}


def write_data_yaml(output_dir: Path) -> Path:
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(output_dir.resolve()), "train": "train/images",
        "val": "val/images", "test": "test/images", "names": TARGET_CLASSES,
    }, sort_keys=False), encoding="utf-8")
    return data_yaml


def detect_source_classes(source_dir: Path) -> Optional[Dict[int, str]]:
    """Try to find class names from data.yaml or classes.txt."""
    yaml_path = source_dir / "data.yaml"
    if yaml_path.exists():
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        names = cfg.get("names", {})
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
        elif isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
    for classes_txt in [source_dir / "classes.txt"] + list(source_dir.rglob("classes.txt"))[:3]:
        if classes_txt and classes_txt.exists():
            names_list = [l.strip() for l in classes_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            if names_list:
                return {i: name for i, name in enumerate(names_list)}
    return None


def auto_detect_and_convert(source_dir: Path, output_dir: Path, name: str) -> Optional[Path]:
    """Auto-detect annotation format and convert to YOLO layout."""
    print(f"\n[INFO] Converting {name}: {source_dir}")

    if not source_dir.exists():
        print(f"[SKIP] Directory does not exist: {source_dir}")
        return None

    if output_dir.exists() and (output_dir / "data.yaml").exists():
        print(f"[SKIP] Already converted: {output_dir}")
        return output_dir / "data.yaml"

    # Check for VOC XML annotations
    xml_files = list(source_dir.rglob("*.xml"))
    has_xml = len(xml_files) > 5

    # Check for YOLO TXT labels
    source_classes = detect_source_classes(source_dir)
    has_splits = any((source_dir / s / "images").exists() for s in ["train", "val", "valid", "test"])

    images = discover_images(source_dir)
    if not images:
        print(f"[WARN] No images found in {source_dir}")
        return None

    print(f"[INFO] Found {len(images)} images, {len(xml_files)} XMLs, splits={has_splits}, classes={source_classes}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    if has_xml and not has_splits:
        # VOC format — find image-xml pairs
        image_xml_pairs: List[Tuple[Path, Path]] = []
        for img in images:
            xml_candidates = [img.with_suffix(".xml")]
            for ann_dir_name in ["Annotations", "annotations", "labels", "xmls"]:
                xml_candidates.append(img.parent / ann_dir_name / f"{img.stem}.xml")
                xml_candidates.append(img.parent.parent / ann_dir_name / f"{img.stem}.xml")
            xml_path = next((x for x in xml_candidates if x.exists()), None)
            if xml_path:
                image_xml_pairs.append((img, xml_path))

        if not image_xml_pairs:
            print(f"[WARN] No image-annotation pairs for {name}")
            return None

        splits = split_images([p[0] for p in image_xml_pairs])
        xml_map = {img: xml for img, xml in image_xml_pairs}

        for split_name, split_imgs in splits.items():
            img_dir = output_dir / split_name / "images"
            lbl_dir = output_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            labeled = 0
            for img_path in split_imgs:
                shutil.copy2(img_path, img_dir / img_path.name)
                if convert_voc_to_yolo(xml_map[img_path], lbl_dir / f"{img_path.stem}.txt"):
                    labeled += 1
            print(f"  {split_name}: {len(split_imgs)} images, {labeled} labeled")

    elif source_classes is not None:
        # YOLO format with known classes — remap
        if has_splits:
            split_mapping = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}
            for target_split, src_candidates in split_mapping.items():
                src_split = next((s for s in src_candidates if (source_dir / s / "images").exists()), None)
                if src_split is None:
                    continue
                src_images_dir = source_dir / src_split / "images"
                src_labels_dir = source_dir / src_split / "labels"
                dst_images = output_dir / target_split / "images"
                dst_labels = output_dir / target_split / "labels"
                dst_images.mkdir(parents=True, exist_ok=True)
                dst_labels.mkdir(parents=True, exist_ok=True)
                labeled = 0
                for img_path in discover_images(src_images_dir):
                    shutil.copy2(img_path, dst_images / img_path.name)
                    if convert_yolo_labels_with_remap(
                        src_labels_dir / f"{img_path.stem}.txt",
                        dst_labels / f"{img_path.stem}.txt", source_classes
                    ):
                        labeled += 1
                print(f"  {target_split}: {len(discover_images(src_images_dir))} images, {labeled} labeled")
        else:
            label_files = {lf.stem: lf for lf in source_dir.rglob("*.txt") if lf.name != "classes.txt"}
            paired = [(img, label_files[img.stem]) for img in images if img.stem in label_files]
            if not paired:
                print(f"[WARN] No image-label pairs for {name}")
                return None
            splits = split_images([p[0] for p in paired])
            lbl_map = {img: lbl for img, lbl in paired}
            for split_name, split_imgs in splits.items():
                dst_images = output_dir / split_name / "images"
                dst_labels = output_dir / split_name / "labels"
                dst_images.mkdir(parents=True, exist_ok=True)
                dst_labels.mkdir(parents=True, exist_ok=True)
                labeled = 0
                for img_path in split_imgs:
                    shutil.copy2(img_path, dst_images / img_path.name)
                    if lbl_map.get(img_path) and convert_yolo_labels_with_remap(
                        lbl_map[img_path], dst_labels / f"{img_path.stem}.txt", source_classes
                    ):
                        labeled += 1
                print(f"  {split_name}: {len(split_imgs)} images, {labeled} labeled")
    else:
        print(f"[WARN] Cannot determine format/classes for {name}")
        return None

    # Ensure all split dirs exist
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    data_yaml = write_data_yaml(output_dir)
    print(f"[OK] Created {data_yaml}")
    return data_yaml


# ── All Kaggle datasets K1-K8 ──────────────────────────────────────────────
KAGGLE_DATASETS = [
    {"name": "K1_aneesarom", "source": "kaggle_aneesarom",
     "slug": "aneesarom/rider-with-helmet-without-helmet-number-plate",
     "priority": "HIGH", "fills": "all violation classes"},
    {"name": "K2_rishabhzen", "source": "kaggle_rishabhzen",
     "slug": "rishabhzen/on-vehicle-helmet-detection",
     "priority": "HIGH", "fills": "surveillance context"},
    {"name": "K3_devgurucodes", "source": "kaggle_devgurucodes",
     "slug": "devgurucodes/trafffic-violations-triple-riding-no-helmet-plate",
     "priority": "CRITICAL", "fills": "triple riding"},
    {"name": "K4_meliodassourav", "source": "kaggle_meliodassourav",
     "slug": "meliodassourav/traffic-violation-dataset-v3",
     "priority": "HIGH", "fills": "overloading / multi-rider"},
    {"name": "K5_hozngvan", "source": "kaggle_hozngvan",
     "slug": "hozngvan/helmet-detection",
     "priority": "HIGH", "fills": "10K motorcycle instances"},
    {"name": "K6_kedarsai", "source": "kaggle_kedarsai",
     "slug": "kedarsai/indian-license-plates-with-labels",
     "priority": "HIGH", "fills": "Indian plates YOLO format"},
    {"name": "K7_guisahanes", "source": "kaggle_guisahanes",
     "slug": "guisahanes/traffic-violation-detection-dataset",
     "priority": "VERIFY", "fills": "ROI-cropped violations"},
    {"name": "K8_andrewmvd", "source": "kaggle_andrewmvd",
     "slug": "andrewmvd/helmet-detection",
     "priority": "HIGH", "fills": "clean helmet/no_helmet bboxes"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Kaggle datasets to YOLO format.")
    p.add_argument("--kaggle-dirs", nargs="*", default=None,
                   help="Specific directories to convert (default: auto-discover)")
    p.add_argument("--download", action="store_true",
                   help="Also download datasets using kaggle CLI")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    converted = 0

    if args.download:
        for ds in KAGGLE_DATASETS:
            dest = KAGGLE_RAW_BASE / ds["source"]
            if dest.exists() and discover_images(dest):
                print(f"[SKIP] {ds['source']} already has images")
                continue
            print(f"\n[DOWNLOAD] {ds['slug']} -> {dest}")
            os.system(f'kaggle datasets download -d {ds["slug"]} -p {dest} --unzip')

    if args.kaggle_dirs:
        for idx, dir_path in enumerate(args.kaggle_dirs):
            source = Path(dir_path)
            if not source.exists():
                continue
            output = KAGGLE_RAW_BASE / f"kaggle_custom_{idx}_yolo"
            if auto_detect_and_convert(source, output, f"custom_{idx}"):
                converted += 1
    else:
        for ds in KAGGLE_DATASETS:
            source = KAGGLE_RAW_BASE / ds["source"]
            output = KAGGLE_RAW_BASE / f"{ds['source']}_yolo"
            if not source.exists():
                print(f"[SKIP] {source} not found. Download: kaggle datasets download -d {ds['slug']} -p {source} --unzip")
                continue
            result = auto_detect_and_convert(source, output, ds["name"])
            if result:
                converted += 1

        # Also check for any other kaggle_* directories
        for kaggle_dir in sorted(KAGGLE_RAW_BASE.glob("kaggle_*")):
            if kaggle_dir.name.endswith("_yolo"):
                continue
            if any(kaggle_dir.name == ds["source"] for ds in KAGGLE_DATASETS):
                continue
            output = KAGGLE_RAW_BASE / f"{kaggle_dir.name}_yolo"
            result = auto_detect_and_convert(kaggle_dir, output, kaggle_dir.name)
            if result:
                converted += 1

    print(f"\n[DONE] Converted {converted} Kaggle dataset(s)")
    if converted > 0:
        print("[NEXT] Run: python download_models.py --merge-only")


if __name__ == "__main__":
    main()
