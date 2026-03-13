from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VisionResult:
    vehicle_count: int
    density: float
    quality_score: float
    boxes: list[tuple[int, int, int, int]]


class VisionService:
    def analyze(self, image_path: str) -> VisionResult:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [cnt for cnt in contours if cv2.contourArea(cnt) > 250]
        boxes: list[tuple[int, int, int, int]] = [cv2.boundingRect(cnt) for cnt in candidates]
        vehicle_count = len(boxes)

        density = min(vehicle_count / 80.0, 1.0)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_score = float(np.clip(laplacian / 300.0, 0.0, 1.0))

        return VisionResult(vehicle_count=vehicle_count, density=density, quality_score=quality_score, boxes=boxes)

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
