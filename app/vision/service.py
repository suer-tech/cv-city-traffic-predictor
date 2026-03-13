from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VisionResult:
    vehicle_count: int
    density: float
    quality_score: float


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
        vehicle_count = len(candidates)

        density = min(vehicle_count / 80.0, 1.0)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_score = float(np.clip(laplacian / 300.0, 0.0, 1.0))

        return VisionResult(vehicle_count=vehicle_count, density=density, quality_score=quality_score)
