# storage

Назначение модуля:
- Вход: сущности домена (camera/metric/prediction/route/external observation).
- Выход: idempotent сохранение и выборки из PostgreSQL.

Ключевые свойства:
- upsert для метрик/прогнозов,
- поддержка route-агрегации,
- выборка последних внешних наблюдений.
