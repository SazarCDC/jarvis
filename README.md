# Jarvis Desktop Assistant (Ollama + Windows)

## Что изменено

- Добавлен единый конфиг `settings.json` (рядом с `main.py`) + шаблон `settings.example.json`.
- Реализован `ConfigLoader` с автодетектом `project_root`, резолвом относительных путей и ENV override.
- Добавлены action-инструменты: `audio`, `web_search`, `web_fetch`, `web_extract`, `monitor`.
- `assistant_core` переведён на роутер + агентный цикл с полем `continue` и ограничением шагов.
- Добавлен resumable-flow: если агент задаёт вопрос (`ask_user`), задача продолжится после ответа пользователя.
- Улучшен контекст: хвост истории + summary длинного диалога.
- В `voice` добавлен follow-up режим: после диалога ассистент слушает несколько секунд без wake word.

## Конфиг

Скопируй пример и отредактируй под свой ПК:

```bash
copy settings.example.json settings.json
```

Ключевые пути (можно относительные):

```json
{
  "paths": {
    "tools_dir": "tools",
    "models_dir": "models",
    "piper_exe": "tools/piper/piper.exe",
    "piper_model": "models/ru_RU-denis-medium.onnx",
    "porcupine_ppn": "models/jarvis.ppn"
  }
}
```

Относительные пути резолвятся от папки проекта (где `main.py`). Это позволяет переносить проект на другой ПК как папку целиком (`tools/`, `models/`, `settings.json`).

ENV-переменные работают как override, но теперь не являются единственным источником конфигурации.

## Запуск

```bash
pip install -r requirements.txt
python main.py
```

## Audio action

`audio` использует модуль PowerShell `AudioDeviceCmdlets`.
Если модуль не установлен, Jarvis возвращает честную ошибку `AUDIO_MODULE_MISSING` и подсказку установки:

```powershell
Install-Module AudioDeviceCmdlets -Scope CurrentUser
```

## Web actions

- `web_search` — DuckDuckGo API (`duckduckgo-search`).
- `web_fetch` — HTTP загрузка страницы.
- `web_extract` — извлечение основного текста (`trafilatura`).

Это позволяет ассистенту не только открыть браузер, но и получить факты программно.
