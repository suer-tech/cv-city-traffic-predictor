# CV City Traffic Predictor (Functional MVP)

Приложение для:
- регулярного захвата кадров камер,
- подсчёта машин и расчёта метрик,
- прогноза на горизонтах 10/20/30 минут по камере и маршруту,
- ручного администрирования камер/ROI/маршрутов через web UI.

## Быстрый старт
1. `cp .env.example .env`
2. `docker compose up --build`
3. Открыть:
   - Admin UI: `http://localhost:8000/`
   - OpenAPI: `http://localhost:8000/docs`
   - MLflow: `http://localhost:5000`

## Что уже реализовано
- API и UI для камер и маршрутов.
- Планировщик и ручной запуск jobs (`/jobs/run-once`, `/jobs/capture`, `/jobs/inference`, `/jobs/retrain`).
- Хранение 5-минутных метрик и прогнозов с idempotent upsert.
- Пороговые оценки congestion probability.
- Почасовое хранение диагностических скриншотов.
- Внешний provider-заглушка для Яндекс-факта (не блокирует pipeline).

## Базовые API
- `GET /dashboard/summary`
- `GET/POST /cameras`
- `PATCH /cameras/{camera_id}/toggle`
- `PATCH /cameras/{camera_id}/roi`
- `GET/POST /routes`
- `GET /metrics?camera_id=<id>&limit=50`
- `GET /predictions?entity_type=camera&entity_id=<id>`

## Инициализация БД
`python scripts/init_db.py`

