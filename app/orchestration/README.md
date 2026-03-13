# orchestration

Назначение модуля:
- Вход: расписание jobs и данные из БД.
- Выход: обновленные метрики, прогнозы и внешние наблюдения.

Job-ы:
- `capture_and_extract_job`
- `inference_job`
- `retrain_job`
- `run_pipeline_once`
