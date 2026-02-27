# Jarvis (Ollama + Windows Desktop)

Персональный AI-ассистент с архитектурой **LLM = мозг / код = исполнитель**.

## Возможности

- Диалоговый интерфейс в одном окне (история, ввод, Execute/Send, STOP, статусы `Idle/Listening/Heard wake word/Thinking/Acting/Speaking`).
- Voice mode: фоновое прослушивание wake word «джарвис» и озвучивание ответов.
- Подключение к локальной Ollama по HTTP API (`/api/chat`).
- Health-check при старте (HEAD `/` и fallback на `GET /api/tags`) с понятной ошибкой.
- LLM всегда должна возвращать JSON-решение:
  - `intent`: `chat | question | action | noise`
  - `thought`, `confidence`, `ask_user`, `response`
  - `actions[]` для выполнения инструментами
  - `memory_update` для долговременной памяти
- Исполнитель action'ов на ПК (Windows):
  - `cmd`, `powershell`, запуск `.bat`
  - `launch` (exe, shell-uri, папка/файл)
  - `search`, `read_file`, `write_file`
  - `keyboard`, `mouse`, `window`
  - `screenshot`, `clipboard`, `browser`, `wait`
- Логирование в JSONL: input, JSON от LLM, actions, результаты, ошибки, voice-события.
- Память в файлах:
  - `jarvis_assistant/memory/facts.json`
  - `jarvis_assistant/memory/preferences.json`

## Быстрый старт

1. Убедись, что Ollama запущена:

```bash
ollama serve
```

2. Установи зависимости:

```bash
pip install -r requirements.txt
```

3. (Опционально) настрой env:

- `OLLAMA_HOST` (по умолчанию `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (по умолчанию `qwen2.5:7b-instruct`)
- `JARVIS_LOG_DIR` (по умолчанию `logs`)
- `JARVIS_MEMORY_DIR` (по умолчанию `jarvis_assistant/memory`)
- `JARVIS_MAX_HISTORY` (по умолчанию `20`)
- `JARVIS_COMMAND_TIMEOUT` (по умолчанию `120`)

4. Запуск:

```bash
python main.py
```

## Voice mode

- Включается кнопкой `Voice: ON` в верхней панели.
- Wake word: `джарвис` (также распознаются `джарвиз`, `жарвис`, `Jarvis`).
- После wake word ассистент говорит «Слушаю», записывает команду и отправляет её в стандартный текстовый pipeline.
- Ответ одновременно появляется в чате и озвучивается через `pyttsx3` (оффлайн TTS).
- Если ассистент уже выполняет задачу, на wake word он отвечает «Подожди секунду».
- Кнопка `STOP` прерывает текущий pipeline, voice-listening и текущую озвучку.

### Ограничения STT

- По умолчанию используется `speech_recognition` + Google Web Speech (`ru-RU`), поэтому нужен интернет.
- При проблемах с микрофоном или сетью ошибка отображается в чате `System`, а voice loop автоматически останавливается.

## Архитектура

- `main.py` — сборка приложения + startup health-check.
- `jarvis_assistant/ollama_client.py` — системный prompt + запрос к `/api/chat`.
- `jarvis_assistant/assistant_core.py` — цикл: input → llm json → execute actions → ответ.
- `jarvis_assistant/executor.py` — набор инструментов управления ПК.
- `jarvis_assistant/ui.py` — Tkinter UI + background-thread без freeze + интеграция voice loop.
- `jarvis_assistant/voice.py` — wake word listener (STT) + TTS + state machine.
- `jarvis_assistant/memory_store.py` — долговременная память (`facts/preferences`).
- `jarvis_assistant/logger.py` — JSONL-логгер.

## Важно

- Безопасность намеренно минимально ограничена: при сомнениях ассистент должен спрашивать пользователя.
- Любое реальное действие на ПК исполняется только кодом через `actions` из JSON от LLM.

## Manual checklist

- "Открой блокнот" → открывается `notepad.exe`.
- "Закрой блокнот" → окно закрывается (window close или fallback `taskkill`).
- "Найди котиков в интернете" → открывается браузер с поиском картинок.
- "Сколько времени" → ассистент отвечает точным текущим временем без шаблонов.
- "Какая погода в Липецке" → открывается браузер со страницей прогноза.
- После выдачи вариантов ответ `1`/`2`/`3` выбирает соответствующее действие.

## Voice checklist

- [ ] Сказать «Джарвис» → слышим «Слушаю» → сказать «Открой блокнот» → блокнот открылся.
- [ ] «Джарвис, который час?» → ответ озвучен и отображён в чате.
- [ ] Ошибка микрофона/нет доступа → понятное сообщение в `System` + `Voice: OFF`.
- [ ] `STOP` во время выполнения action’ов → останавливает executor + UI возвращается в `Idle`.
- [ ] `Voice: OFF` → фоновое прослушивание отключено, текстовый режим работает как раньше.
