# capture

Назначение модуля:
- Вход: `camera_id`, `source_url`.
- Выход: `CaptureResult` с путем к кадру, временем, провайдером, признаком hourly persistence.

Особенности:
- `capture_provider=playwright|synthetic`.
- retry определяется `CAPTURE_RETRY_COUNT`.
- 1 кадр в час сохраняется в `hourly`, промежуточные — `ephemeral`.
