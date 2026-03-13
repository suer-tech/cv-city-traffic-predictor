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
- При ручном запуске capture сохраняется аннотированный кадр с bounding-box авто (по умолчанию из реального Playwright-захвата).
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


## Проверка детекции вручную
1. Нажмите в Admin UI кнопку `Run capture+detect once` или `Run full pipeline once`.
2. Система сохранит кадр с боксами в `data/snapshots/annotated/...`.
3. Смотреть можно прямо в UI (блок `Latest annotated detection`) или через API `GET /debug/annotated`.


## Важно про реальный кадр
- По умолчанию включен `CAPTURE_PROVIDER=playwright` и `CAPTURE_ALLOW_SYNTHETIC_FALLBACK=false`, то есть фейковый кадр не подставляется тихо.
- Если Playwright недоступен, захват завершится ошибкой (как и должно быть для честной проверки).
- Включайте `CAPTURE_ALLOW_SYNTHETIC_FALLBACK=true` только для локальной отладки интерфейса.
