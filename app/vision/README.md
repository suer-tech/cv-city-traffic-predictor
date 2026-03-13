# vision

Назначение модуля:
- Вход: путь к изображению (скриншот камеры).
- Выход: `VisionResult` — vehicle_count, density, quality_score, boxes (боксы каждого авто).

Детекция:
- YOLOv8 (Ultralytics) с моделью COCO — классы car, motorcycle, bus, truck.
- Конфиг: `VISION_MODEL`, `VISION_CONFIDENCE_THRESHOLD`.
