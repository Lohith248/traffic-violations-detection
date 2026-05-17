from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from paddleocr import PaddleOCR


Box = Tuple[float, float, float, float]

_STANDARD_TASK_ORDER = (
    "motorcycle",
    "person",
    "helmet",
    "no_helmet",
    "license_plate",
)


@dataclass
class Detection:
    box: Box
    confidence: float
    class_id: int
    class_name: str


# =============================================================================
# TrafficViolationDetector — Main submission class (AID 728, Team 7)
#
# Two-stage pipeline:
#   Stage 1: YOLO26s detects 5 classes (motorcycle, person, helmet,
#            no_helmet, license_plate) from a single RGB image.
#   Stage 2: Spatial heuristics associate riders to motorcycles,
#            check helmet compliance, and invoke PaddleOCR on plates
#            of violating vehicles only.
# =============================================================================
class TrafficViolationDetector:
    def __init__(self, model_dir: str = "./models") -> None:
        """
        Initialize and load all models.

        Args:
            model_dir: PaddleOCR model assets directory (submission layout stays ./models/...).

        YOLO weights resolution (first hit wins):
          - TRAFFIC_YOLO_WEIGHTS or YOLO_WEIGHTS env var
          - ./actual_weights/best.pt (or last.pt)
          - ./models/yolo26s_traffic.pt / yolo26m_traffic.pt (fine-tuned YOLO26)
          - ./models/yolo11s_traffic.pt / yolo11m_traffic.pt (fallback)
          - ./models/yolov8s_traffic.pt / yolov8m_traffic.pt (legacy)
          - ./h/*.pt and ./models/yolo26*.pt / yolo11*.pt / yolov8*.pt
        """
        try:
            self.model_dir = Path(model_dir)
            self.max_ocr_attempts = 4
            self.person_motorcycle_iou_threshold = 0.15
            self.head_region_ratio = 0.35
            self.detector_conf_threshold = 0.20
            self.detector_iou_threshold = 0.45
            self.inference_time_budget_ms = 5000.0
            self.ocr_soft_budget_ms = 4300.0
            self.min_plate_overlap_ratio = 0.05

            self.yolo_checkpoint_path: Optional[str] = None
            self.yolo_model: Optional[YOLO] = None
            self.ocr_model: Optional[PaddleOCR] = None
            self._class_id_to_bucket: Dict[int, Optional[str]] = {}

            self._load_models()
        except Exception:
            self.yolo_model = None
            self.ocr_model = None
            self.yolo_checkpoint_path = None

    @staticmethod
    def _solution_root() -> Path:
        return Path(__file__).resolve().parent

    def _resolve_yolo_weights(self) -> Path:
        env = (os.environ.get("TRAFFIC_YOLO_WEIGHTS") or os.environ.get("YOLO_WEIGHTS") or "").strip()
        if env:
            p = Path(os.path.expandvars(os.path.expanduser(env))).resolve()
            if p.is_file():
                return p

        root = self._solution_root()
        ordered = [
            root / "actual_weights" / "best.pt",
            root / "actual_weights" / "last.pt",
            self.model_dir / "yolo26s_traffic.pt",
            self.model_dir / "yolo26m_traffic.pt",
            self.model_dir / "yolo11s_traffic.pt",
            self.model_dir / "yolo11m_traffic.pt",
            self.model_dir / "yolov8s_traffic.pt",
            self.model_dir / "yolov8m_traffic.pt",
            root / "h" / "yolo26s.pt",
            root / "h" / "yolo26m.pt",
            self.model_dir / "yolo26s.pt",
            self.model_dir / "yolo26m.pt",
            root / "h" / "yolo11s.pt",
            root / "h" / "yolo11m.pt",
            self.model_dir / "yolo11s.pt",
            self.model_dir / "yolo11m.pt",
            root / "h" / "yolov8s.pt",
            root / "h" / "yolov8m.pt",
            self.model_dir / "yolov8s.pt",
            self.model_dir / "yolov8m.pt",
        ]
        for cand in ordered:
            if cand.is_file():
                return cand.resolve()
        raise FileNotFoundError(
            "No YOLO weights found. Put best.pt in ./actual_weights/ or set TRAFFIC_YOLO_WEIGHTS, "
            "or use ./models/yolo26s_traffic.pt / export_weights.py after training."
        )

    def _normalize_label_tokens(self, raw: str) -> str:
        return raw.strip().lower().replace("-", "_").replace(" ", "_")

    def _bucket_for_label_text(self, raw: str) -> Optional[str]:
        n = self._normalize_label_tokens(raw)
        tokens = frozenset(t for t in n.split("_") if t)

        if n == "bicycle":
            return None

        if n in {"motorcycle", "motorbike", "scooter"}:
            return "motorcycle"

        person_tokens = {"person", "rider", "pillion", "human", "man", "woman", "people", "driver", "passenger"}
        if n == "person" or tokens & person_tokens:
            if "traffic" in tokens:
                return None
            return "person"

        negative_helmet = {"no", "without", "non", "off", "bare"}
        has_helmet_word = "helmet" in n
        if has_helmet_word and tokens & negative_helmet:
            return "no_helmet"
        if has_helmet_word and tokens & {"plate", "licence", "license"}:
            return None
        if has_helmet_word:
            return "helmet"

        plate_tokens = {"plate", "licence", "license", "registration", "numberplate"}
        if tokens & plate_tokens or any(k in n for k in ["licenceplate", "licenseplate", "number_plate"]):
            return "license_plate"

        return None

    def _refresh_class_buckets(self) -> None:
        self._class_id_to_bucket = {}
        if self.yolo_model is None:
            return
        try:
            names = getattr(self.yolo_model, "names", {})
            ordered: Dict[int, str] = {}
            if isinstance(names, dict):
                for k, v in names.items():
                    ordered[int(k)] = str(v)
            elif isinstance(names, (list, tuple)):
                ordered = {i: str(label) for i, label in enumerate(names)}

            seq = tuple(_STANDARD_TASK_ORDER)
            normalized_seq_match = False
            if len(ordered) == len(seq):
                normalized_seq_match = tuple(
                    self._normalize_label_tokens(ordered[i]) for i in range(len(seq))
                ) == seq

            for class_id in sorted(ordered.keys()):
                label = ordered[class_id]
                bucket = self._bucket_for_label_text(label)
                if bucket is None and normalized_seq_match:
                    bucket = seq[class_id] if 0 <= class_id < len(seq) else None
                self._class_id_to_bucket[class_id] = bucket
        except Exception:
            self._class_id_to_bucket = {}

    def _load_models(self) -> None:
        try:
            yolo_weights = self._resolve_yolo_weights()
            self.yolo_checkpoint_path = str(yolo_weights.resolve())
            self.yolo_model = YOLO(str(yolo_weights))
            self._refresh_class_buckets()

            det_candidates = [
                self.model_dir / "paddle_det",
                self.model_dir / "paddleocr" / "det" / "ch_PP-OCRv4_det_infer",
            ]
            rec_candidates = [
                self.model_dir / "paddle_rec",
                self.model_dir / "paddleocr" / "rec" / "en_PP-OCRv4_rec_infer",
            ]
            cls_candidates = [
                self.model_dir / "paddle_cls",
                self.model_dir / "paddleocr" / "cls" / "ch_ppocr_mobile_v2.0_cls_infer",
            ]

            root_dir = self._solution_root()
            det_candidates.extend(
                [
                    root_dir / "paddle_det",
                    root_dir / "paddleocr" / "det" / "ch_PP-OCRv4_det_infer",
                ]
            )
            rec_candidates.extend(
                [
                    root_dir / "paddle_rec",
                    root_dir / "paddleocr" / "rec" / "en_PP-OCRv4_rec_infer",
                ]
            )
            cls_candidates.extend(
                [
                    root_dir / "paddle_cls",
                    root_dir / "paddleocr" / "cls" / "ch_ppocr_mobile_v2.0_cls_infer",
                ]
            )

            det_dir = next((p for p in det_candidates if self._is_valid_paddle_dir(p)), None)
            rec_dir = next((p for p in rec_candidates if self._is_valid_paddle_dir(p)), None)
            cls_dir = next((p for p in cls_candidates if self._is_valid_paddle_dir(p)), None)

            if det_dir is not None and rec_dir is not None and cls_dir is not None:
                self.ocr_model = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=False,
                    show_log=False,
                    det_model_dir=str(det_dir),
                    rec_model_dir=str(rec_dir),
                    cls_model_dir=str(cls_dir),
                )
            else:
                self.ocr_model = None
        except Exception:
            self.yolo_model = None
            self.ocr_model = None
            self.yolo_checkpoint_path = None

    def _is_valid_paddle_dir(self, directory: Path) -> bool:
        try:
            return (
                directory.exists()
                and (directory / "inference.pdmodel").exists()
                and (directory / "inference.pdiparams").exists()
            )
        except Exception:
            return False

    # ── PUBLIC API: Stateless single-image inference ──────────────────────
    def predict(self, image_path: str) -> dict:
        """
        Run stateless single-image inference and return only violating motorcycles.

        Args:
            image_path: Path to the input image.

        Returns:
            A JSON-compatible dictionary in the required format:
            {
              "violations": [
                {
                  "num_riders": int,
                  "helmet_violations": int,
                  "license_plate": str
                }
              ]
            }
        """
        try:
            if self.yolo_model is None:
                return {"violations": []}

            start_time = time.perf_counter()
            image = self._load_image(image_path)
            if image is None:
                return {"violations": []}

            detections = self._detect(image)
            motorcycles = sorted(
                detections["motorcycle"],
                key=lambda d: (-d.confidence, d.box[0], d.box[1], d.box[2], d.box[3]),
            )
            persons = detections["person"]
            helmets = detections["helmet"]
            license_plates = detections["license_plate"]

            if not motorcycles:
                return {"violations": []}

            violations: List[dict] = []
            ocr_attempts = 0

            # --- Core violation logic: per-motorcycle assessment ---
            for motorcycle in motorcycles:
                riders = self._get_riders_for_motorcycle(motorcycle, persons)
                num_riders = len(riders)
                helmet_violations = self._count_helmet_violations(riders, helmets)

                # Only report violating vehicles; skip compliant ones
                if not ((num_riders > 2) or (helmet_violations > 0)):
                    continue

                plate_text = ""
                if self._can_run_ocr(ocr_attempts, start_time):
                    best_plate = self._select_best_plate_for_motorcycle(motorcycle, license_plates)
                    if best_plate is not None:
                        plate_text, used_attempts = self._read_plate_text(
                            image,
                            best_plate.box,
                            remaining_attempts=self.max_ocr_attempts - ocr_attempts,
                        )
                        ocr_attempts += used_attempts

                violations.append(
                    {
                        "num_riders": num_riders,
                        "helmet_violations": helmet_violations,
                        "license_plate": plate_text,
                    }
                )

            return {"violations": violations}
        except Exception:
            return {"violations": []}

    def _can_run_ocr(self, ocr_attempts: int, start_time: float) -> bool:
        try:
            if ocr_attempts >= self.max_ocr_attempts:
                return False
            return self._elapsed_ms(start_time) < self.ocr_soft_budget_ms
        except Exception:
            return False

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = cv2.imread(image_path)
            return image
        except Exception:
            return None

    def _detect(self, image: np.ndarray) -> Dict[str, List[Detection]]:
        try:
            grouped: Dict[str, List[Detection]] = {
                "motorcycle": [],
                "person": [],
                "helmet": [],
                "no_helmet": [],
                "license_plate": [],
            }
            if self.yolo_model is None:
                return grouped

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # YOLO26 is NMS-free end-to-end; Ultralytics still returns results[0].boxes the same way.
            # The iou= argument may be ignored for YOLO26 but is kept for YOLO11/YOLOv8 checkpoints.
            results = self.yolo_model.predict(
                source=rgb,
                conf=self.detector_conf_threshold,
                iou=self.detector_iou_threshold,
                imgsz=640,
                verbose=False,
            )
            if not results:
                return grouped

            parsed = self._parse_result(results[0])
            for det in parsed:
                target = self._map_model_class_to_target(det.class_id, det.class_name)
                if target is not None and target in grouped:
                    grouped[target].append(det)

            for key in grouped:
                grouped[key].sort(key=lambda d: d.confidence, reverse=True)
            return grouped
        except Exception:
            return {
                "motorcycle": [],
                "person": [],
                "helmet": [],
                "no_helmet": [],
                "license_plate": [],
            }

    def _parse_result(self, result: Any) -> List[Detection]:
        try:
            detections: List[Detection] = []
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return detections

            names = result.names if hasattr(result, "names") else {}
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            for idx in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[idx].tolist()
                class_id = int(cls[idx])
                class_name = str(names.get(class_id, "")).strip().lower()
                detections.append(
                    Detection(
                        box=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf[idx]),
                        class_id=class_id,
                        class_name=class_name,
                    )
                )
            return detections
        except Exception:
            return []

    def _map_model_class_to_target(self, class_id: int, class_name: str) -> Optional[str]:
        try:
            bucket = self._class_id_to_bucket.get(class_id)
            if bucket is not None:
                return bucket
            return self._bucket_for_label_text(class_name)
        except Exception:
            return None

    # ── SPATIAL HEURISTICS: Rider-to-motorcycle assignment via IoU ─────
    def _get_riders_for_motorcycle(self, motorcycle: Detection, persons: List[Detection]) -> List[Detection]:
        try:
            riders: List[Detection] = []
            for person in persons:
                if self._iou(motorcycle.box, person.box) > self.person_motorcycle_iou_threshold:
                    riders.append(person)
            return riders
        except Exception:
            return []

    def _has_box_centroid_in_head_region(self, rider_box: Box, boxes: List[Detection]) -> bool:
        px1, py1, px2, py2 = rider_box
        top_y = py1 + self.head_region_ratio * max(0.0, (py2 - py1))
        for item in boxes:
            cx, cy = self._centroid(item.box)
            if px1 <= cx <= px2 and py1 <= cy <= top_y:
                return True
        return False

    # ── HELMET CHECK: Top 35% of rider bbox checked for helmet centroid ─
    def _count_helmet_violations(self, riders: List[Detection], helmets: List[Detection]) -> int:
        try:
            violations = 0
            for rider in riders:
                if not self._has_helmet_in_head_region(rider.box, helmets):
                    violations += 1
            return violations
        except Exception:
            return 0

    def _has_helmet_in_head_region(self, rider_box: Box, helmets: List[Detection]) -> bool:
        try:
            return self._has_box_centroid_in_head_region(rider_box, helmets)
        except Exception:
            return False

    def _select_best_plate_for_motorcycle(
        self, motorcycle: Detection, plates: List[Detection]
    ) -> Optional[Detection]:
        try:
            best: Optional[Detection] = None
            best_score = -1.0
            for plate in plates:
                plate_area = self._area(plate.box)
                if plate_area <= 0.0:
                    continue
                overlap_ratio = self._intersection_area(motorcycle.box, plate.box) / plate_area
                if overlap_ratio < self.min_plate_overlap_ratio:
                    continue

                score = (0.65 * plate.confidence) + (0.35 * overlap_ratio)
                if score > best_score:
                    best = plate
                    best_score = score
            return best
        except Exception:
            return None

    # ── OCR PIPELINE: Multi-attempt PaddleOCR with preprocessing ────────
    def _read_plate_text(
        self, image: np.ndarray, plate_box: Box, remaining_attempts: int
    ) -> Tuple[str, int]:
        try:
            if self.ocr_model is None or remaining_attempts <= 0:
                return ("", 0)

            crop = self._crop_box(image, plate_box)
            if crop is None or crop.size == 0:
                return ("", 0)

            variants = self._build_plate_ocr_variants(crop)
            if not variants:
                return ("", 0)

            attempts_used = 0
            best_text = ""
            best_score = -1.0
            max_runs = min(remaining_attempts, len(variants))

            for idx in range(max_runs):
                result = self.ocr_model.ocr(variants[idx], cls=True)
                attempts_used += 1
                text, score = self._extract_plate_text_with_score(result)
                if (score > best_score) or (
                    score == best_score and len(text) > len(best_text)
                ):
                    best_text = text
                    best_score = score

            return (best_text, attempts_used)
        except Exception:
            return ("", 0)

    def _build_plate_ocr_variants(self, plate_crop: np.ndarray) -> List[np.ndarray]:
        try:
            variants: List[np.ndarray] = []
            first = self._preprocess_plate_for_ocr(plate_crop, use_clahe=False)
            second = self._preprocess_plate_for_ocr(plate_crop, use_clahe=True)
            if first is not None and first.size > 0:
                variants.append(first)
            if second is not None and second.size > 0:
                variants.append(second)
            return variants
        except Exception:
            return []

    def _crop_box(self, image: np.ndarray, box: Box) -> Optional[np.ndarray]:
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = box

            x1_i = max(0, min(w - 1, int(round(x1))))
            y1_i = max(0, min(h - 1, int(round(y1))))
            x2_i = max(0, min(w, int(round(x2))))
            y2_i = max(0, min(h, int(round(y2))))

            if x2_i <= x1_i or y2_i <= y1_i:
                return None
            return image[y1_i:y2_i, x1_i:x2_i].copy()
        except Exception:
            return None

    def _preprocess_plate_for_ocr(
        self, plate_crop: np.ndarray, use_clahe: bool
    ) -> Optional[np.ndarray]:
        try:
            processed = plate_crop
            h, w = processed.shape[:2]
            if w < 120:
                scale = 120.0 / max(w, 1)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            processed = cv2.filter2D(processed, ddepth=-1, kernel=sharpen_kernel)

            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return bgr
        except Exception:
            return None

    def _extract_plate_text_with_score(self, ocr_result: Any) -> Tuple[str, float]:
        try:
            if not ocr_result:
                return ("", 0.0)

            pieces: List[str] = []
            scores: List[float] = []
            for item in ocr_result:
                if not item:
                    continue
                for line in item:
                    if (
                        isinstance(line, (list, tuple))
                        and len(line) >= 2
                        and isinstance(line[1], (list, tuple))
                        and len(line[1]) >= 1
                    ):
                        text = str(line[1][0]).strip()
                        confidence = (
                            float(line[1][1])
                            if len(line[1]) >= 2 and isinstance(line[1][1], (float, int))
                            else 0.0
                        )
                        if text:
                            pieces.append(text)
                            scores.append(confidence)

            if not pieces:
                return ("", 0.0)

            merged = "".join(pieces).upper().strip()
            merged = re.sub(r"\s+", "", merged)
            if not merged:
                return ("", 0.0)

            avg_score = float(sum(scores) / len(scores)) if scores else 0.0
            return (merged, avg_score)
        except Exception:
            return ("", 0.0)

    def _centroid(self, box: Box) -> Tuple[float, float]:
        try:
            x1, y1, x2, y2 = box
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        except Exception:
            return (0.0, 0.0)

    def _intersection_area(self, box_a: Box, box_b: Box) -> float:
        try:
            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            return float((ix2 - ix1) * (iy2 - iy1))
        except Exception:
            return 0.0

    def _area(self, box: Box) -> float:
        try:
            x1, y1, x2, y2 = box
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)
        except Exception:
            return 0.0

    def _elapsed_ms(self, start_time: float) -> float:
        try:
            return (time.perf_counter() - start_time) * 1000.0
        except Exception:
            return self.inference_time_budget_ms

    def _iou(self, box_a: Box, box_b: Box) -> float:
        try:
            inter = self._intersection_area(box_a, box_b)
            if inter <= 0.0:
                return 0.0

            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            if union <= 0.0:
                return 0.0
            return float(inter / union)
        except Exception:
            return 0.0

