from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import settings

# COCO class IDs for vehicles: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = (2, 3, 5, 7)


@dataclass
class VisionResult:
    vehicle_count: int
    density: float
    quality_score: float
    boxes: list[tuple[int, int, int, int]]


class VisionService:
    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(settings.vision_model)
        return self._model

    def analyze(self, image_path: str) -> VisionResult:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        model = self._get_model()
        results = model.predict(
            image_path,
            conf=settings.vision_confidence_threshold,
            verbose=False,
        )

        boxes: list[tuple[int, int, int, int]] = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for i, cls_id in enumerate(r.boxes.cls.int().tolist()):
                    if cls_id in VEHICLE_CLASS_IDS:
                        x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                        x, y = int(x1), int(y1)
                        w, h = int(x2 - x1), int(y2 - y1)
                        boxes.append((x, y, w, h))

        vehicle_count = len(boxes)
        density = min(vehicle_count / 80.0, 1.0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_score = float(np.clip(laplacian / 300.0, 0.0, 1.0))

        return VisionResult(
            vehicle_count=vehicle_count,
            density=density,
            quality_score=quality_score,
            boxes=boxes,
        )

    def save_annotated(self, image_path: str, boxes: list[tuple[int, int, int, int]], output_path: str) -> str:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        for x, y, w, h in boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.putText(
            image,
            f"vehicles={len(boxes)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), image)
        return str(out)
