from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import yaml
from ultralytics import YOLO


def build_indian_road_augmenter() -> A.Compose:
    """
    Albumentations config tuned for Indian road conditions:
    rain, fog, motion blur, brightness variance, and compression artifacts.
    """
    return A.Compose(
        [
            A.OneOf(
                [
                    A.RandomRain(
                        slant_lower=-15,
                        slant_upper=15,
                        drop_length=16,
                        drop_width=1,
                        blur_value=3,
                        brightness_coefficient=0.9,
                        p=1.0,
                    ),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.35, alpha_coef=0.08, p=1.0),
                ],
                p=0.45,
            ),
            A.MotionBlur(blur_limit=(3, 7), p=0.30),
            A.RandomBrightnessContrast(
                brightness_limit=0.30,
                contrast_limit=0.25,
                p=0.50,
            ),
            A.ImageCompression(quality_lower=35, quality_upper=95, p=0.40),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.15,
        ),
    )


def load_data_yaml(data_yaml_path: Path) -> Dict:
    return yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))


def write_data_yaml(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def list_image_files(directory: Path) -> List[Path]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in image_suffixes])


def read_yolo_labels(label_file: Path, num_classes: int) -> Tuple[List[int], List[List[float]]]:
    if not label_file.exists():
        return [], []

    class_ids: List[int] = []
    bboxes: List[List[float]] = []
    for raw in label_file.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = [float(v) for v in parts[1:]]
        except ValueError:
            continue

        if cls < 0 or cls >= num_classes:
            continue
        if bw <= 0.0 or bh <= 0.0:
            continue

        xc = min(1.0, max(0.0, xc))
        yc = min(1.0, max(0.0, yc))
        bw = min(1.0, max(1e-6, bw))
        bh = min(1.0, max(1e-6, bh))

        class_ids.append(cls)
        bboxes.append([xc, yc, bw, bh])
    return class_ids, bboxes


def write_yolo_labels(label_file: Path, class_ids: List[int], bboxes: List[List[float]]) -> None:
    label_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls_id} {' '.join(f'{v:.6f}' for v in box)}" for cls_id, box in zip(class_ids, bboxes)]
    label_file.write_text("\n".join(lines), encoding="utf-8")


def copy_split_sanitized(
    source_images: Path,
    source_labels: Path,
    dest_images: Path,
    dest_labels: Path,
    num_classes: int,
) -> None:
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    for image_path in list_image_files(source_images):
        shutil.copy2(image_path, dest_images / image_path.name)
        src_label = source_labels / f"{image_path.stem}.txt"
        dst_label = dest_labels / f"{image_path.stem}.txt"
        class_ids, boxes = read_yolo_labels(src_label, num_classes=num_classes)
        write_yolo_labels(dst_label, class_ids, boxes)


def resolve_split_paths(dataset_root: Path, split_images_rel: str) -> Tuple[Path, Path]:
    images_dir = dataset_root / Path(split_images_rel)
    labels_dir = images_dir.parent / "labels"
    return images_dir, labels_dir


def collect_class_counts(
    train_images: Path,
    train_labels: Path,
    num_classes: int,
) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    for image_path in list_image_files(train_images):
        label_path = train_labels / f"{image_path.stem}.txt"
        class_ids, _ = read_yolo_labels(label_path, num_classes=num_classes)
        for class_id in class_ids:
            counts[class_id] += 1
    return counts


def quantile_value(values: Sequence[int], q: float) -> int:
    non_zero = sorted([v for v in values if v > 0])
    if not non_zero:
        return 1
    idx = int(round((len(non_zero) - 1) * q))
    idx = max(0, min(len(non_zero) - 1, idx))
    return max(1, non_zero[idx])


def build_class_balanced_dataset(
    source_data_yaml: Path,
    output_root: Path,
    max_repeats: int,
) -> Path:
    """
    Builds a class-balanced dataset by oversampling train images containing
    underrepresented classes.
    """
    cfg = load_data_yaml(source_data_yaml)
    names = cfg["names"]
    if isinstance(names, dict):
        num_classes = len(names.keys())
    else:
        num_classes = len(names)

    dataset_root = Path(cfg.get("path", source_data_yaml.parent)).resolve()
    train_images, train_labels = resolve_split_paths(dataset_root, cfg["train"])
    val_images, val_labels = resolve_split_paths(dataset_root, cfg["val"])
    test_images, test_labels = resolve_split_paths(dataset_root, cfg.get("test", cfg["val"]))

    out_train_images = output_root / "train" / "images"
    out_train_labels = output_root / "train" / "labels"
    out_val_images = output_root / "val" / "images"
    out_val_labels = output_root / "val" / "labels"
    out_test_images = output_root / "test" / "images"
    out_test_labels = output_root / "test" / "labels"

    copy_split_sanitized(train_images, train_labels, out_train_images, out_train_labels, num_classes)
    copy_split_sanitized(val_images, val_labels, out_val_images, out_val_labels, num_classes)
    copy_split_sanitized(test_images, test_labels, out_test_images, out_test_labels, num_classes)

    class_counts = collect_class_counts(train_images, train_labels, num_classes=num_classes)
    target_count = quantile_value(class_counts, q=0.70)
    class_weights = [float(target_count / max(1, c)) for c in class_counts]

    for image_path in list_image_files(train_images):
        label_path = train_labels / f"{image_path.stem}.txt"
        class_ids, boxes = read_yolo_labels(label_path, num_classes=num_classes)
        if not class_ids:
            continue

        image_weight = max(class_weights[c] for c in set(class_ids))
        repeats = min(max_repeats, max(1, int(round(image_weight))))

        for rep_idx in range(1, repeats):
            dst_image = out_train_images / f"{image_path.stem}_rep{rep_idx}{image_path.suffix.lower()}"
            dst_label = out_train_labels / f"{image_path.stem}_rep{rep_idx}.txt"
            shutil.copy2(image_path, dst_image)
            write_yolo_labels(dst_label, class_ids, boxes)

    balanced_yaml = output_root / "data.yaml"
    write_data_yaml(
        {
            "path": str(output_root.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": cfg["names"],
        },
        balanced_yaml,
    )
    return balanced_yaml


def prepare_augmented_dataset(
    source_data_yaml: Path,
    output_root: Path,
    copies_per_image: int,
) -> Path:
    """
    Creates an augmented copy of the source YOLO dataset.
    - train split: original + augmented copies
    - val/test splits: copied as-is
    """
    cfg = load_data_yaml(source_data_yaml)
    names = cfg["names"]
    if isinstance(names, dict):
        num_classes = len(names.keys())
    else:
        num_classes = len(names)

    dataset_root = Path(cfg.get("path", source_data_yaml.parent)).resolve()
    train_images, train_labels = resolve_split_paths(dataset_root, cfg["train"])
    val_images, val_labels = resolve_split_paths(dataset_root, cfg["val"])
    test_images, test_labels = resolve_split_paths(dataset_root, cfg.get("test", cfg["val"]))

    out_train_images = output_root / "train" / "images"
    out_train_labels = output_root / "train" / "labels"
    out_val_images = output_root / "val" / "images"
    out_val_labels = output_root / "val" / "labels"
    out_test_images = output_root / "test" / "images"
    out_test_labels = output_root / "test" / "labels"

    copy_split_sanitized(train_images, train_labels, out_train_images, out_train_labels, num_classes)
    copy_split_sanitized(val_images, val_labels, out_val_images, out_val_labels, num_classes)
    copy_split_sanitized(test_images, test_labels, out_test_images, out_test_labels, num_classes)

    augmenter = build_indian_road_augmenter()

    for image_path in list_image_files(train_images):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        label_path = train_labels / f"{image_path.stem}.txt"
        class_ids, bboxes = read_yolo_labels(label_path, num_classes=num_classes)

        for i in range(copies_per_image):
            transformed = augmenter(image=image, bboxes=bboxes, class_labels=class_ids)
            aug_image = transformed["image"]
            aug_boxes = transformed["bboxes"]
            aug_class_ids = transformed["class_labels"]

            aug_stem = f"{image_path.stem}_aug{i + 1}"
            aug_image_path = out_train_images / f"{aug_stem}{image_path.suffix.lower()}"
            aug_label_path = out_train_labels / f"{aug_stem}.txt"

            cv2.imwrite(str(aug_image_path), aug_image)
            write_yolo_labels(aug_label_path, list(aug_class_ids), [list(b) for b in aug_boxes])

    augmented_yaml = output_root / "data.yaml"
    write_data_yaml(
        {
            "path": str(output_root.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": cfg["names"],
        },
        augmented_yaml,
    )
    return augmented_yaml


def train_model(
    merged_data_yaml: Path,
    model_dir: Path,
    base_weights_name: str = "yolov8s.pt",
    output_weights_name: str = "yolov8s_traffic.pt",
    epochs: int = 50,
    imgsz: int = 640,
    device: str = "0",
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)

    base_weights = model_dir / base_weights_name
    if not base_weights.exists():
        raise FileNotFoundError(
            f"Base weights not found at {base_weights}. Run download_models.py first."
        )

    model = YOLO(str(base_weights))
    model.train(
        data=str(merged_data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(model_dir / "runs"),
        name="traffic_violation",
        exist_ok=True,
        pretrained=True,
        deterministic=True,
        seed=42,
        device=device,
    )

    best_path = Path(str(model.trainer.best))
    if not best_path.exists():
        raise FileNotFoundError("Could not locate best.pt after training.")

    destination = model_dir / output_weights_name
    shutil.copy2(best_path, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8s for traffic violation detection.")
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="./datasets/merged/data.yaml",
        help="Path to merged YOLO data.yaml",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Directory that contains yolov8s.pt and where trained weights are saved",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default="yolov8s.pt",
        help="Base Ultralytics weights file name inside model-dir (e.g. yolov8s.pt, yolov8m.pt)",
    )
    parser.add_argument(
        "--output-weights",
        type=str,
        default="yolov8s_traffic.pt",
        help="Output trained weights file name inside model-dir",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Training device for Ultralytics (e.g. 0 for first GPU, cpu for CPU)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument(
        "--rebalance-train",
        action="store_true",
        help="Create class-balanced train split by oversampling underrepresented classes",
    )
    parser.add_argument(
        "--rebalance-output-root",
        type=str,
        default="./datasets/merged_rebalanced",
        help="Output directory for class-balanced dataset",
    )
    parser.add_argument(
        "--max-repeats",
        type=int,
        default=3,
        help="Maximum total copies per train image during class balancing",
    )
    parser.add_argument(
        "--augment-offline",
        action="store_true",
        help="Generate augmented dataset copies with Albumentations before training",
    )
    parser.add_argument(
        "--aug-copies-per-image",
        type=int,
        default=1,
        help="How many augmented copies to create per train image",
    )
    parser.add_argument(
        "--aug-output-root",
        type=str,
        default="./datasets/merged_augmented",
        help="Output directory for offline-augmented dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data_yaml).resolve()
    model_dir = Path(args.model_dir).resolve()

    if args.rebalance_train:
        data_yaml = build_class_balanced_dataset(
            source_data_yaml=data_yaml,
            output_root=Path(args.rebalance_output_root).resolve(),
            max_repeats=max(1, args.max_repeats),
        )
        print(f"[INFO] Using class-balanced dataset: {data_yaml}")

    if args.augment_offline:
        data_yaml = prepare_augmented_dataset(
            source_data_yaml=data_yaml,
            output_root=Path(args.aug_output_root).resolve(),
            copies_per_image=max(1, args.aug_copies_per_image),
        )
        print(f"[INFO] Using augmented dataset: {data_yaml}")

    output_weights = train_model(
        merged_data_yaml=data_yaml,
        model_dir=model_dir,
        base_weights_name=args.base_weights,
        output_weights_name=args.output_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"[DONE] Best model saved to: {output_weights}")


if __name__ == "__main__":
    main()
