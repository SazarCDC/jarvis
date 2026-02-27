# Jarvis (Ollama + Windows Desktop)

Персональный AI-ассистент с архитектурой **LLM = мозг / код = исполнитель**.

## Возможности

- Диалоговый интерфейс в одном окне (история, ввод, Execute/Send, STOP, статусы `Idle/Listening/Heard wake word/Thinking/Acting/Speaking`).
- Полностью оффлайн voice mode: wake word через `openWakeWord` (модель `hey_jarvis`) **только через ONNX backend**, STT через `faster-whisper`, VAD через Silero (`torch`) с fallback на VAD-lite, TTS через `pyttsx3`.
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

## Voice mode (offline)

- Включается кнопкой `Voice: ON` в верхней панели.
- Wake word в UI/логах: `джарвис` (детекция через `openWakeWord` ONNX-модель из локальной папки проекта).
- После wake word ассистент говорит «Слушаю», записывает команду по VAD и отправляет её в стандартный текстовый pipeline.
- Ответ появляется в чате и озвучивается через `pyttsx3`.
- Если ассистент уже выполняет задачу, на wake word он отвечает «Подожди секунду».
- Кнопка `STOP` останавливает текущий pipeline, voice-listening, текущую запись/транскрипцию и текущую озвучку.
- В UI есть слайдер `TTS Volume` (0..100), который применяется сразу.

### Voice ENV

- `JARVIS_WAKE_THRESHOLD` (default `0.65`)
- `JARVIS_WAKE_COOLDOWN` (default `2.0`)
- `JARVIS_WAKE_MODEL_PATH` (optional: путь к конкретному wake-model `*.onnx`; рядом должны лежать `melspectrogram.onnx` и `embedding_model.onnx`)
- `JARVIS_WAKE_MODELS_DIR` (optional: каталог моделей; default `models/openwakeword/`)
- `JARVIS_AUTO_DOWNLOAD_MODELS` (optional: `1|true|yes|on`; разрешает авто-скачивание ONNX моделей с HuggingFace)
- `JARVIS_VAD_BACKEND` (`silero|lite`, default `silero`)
- `JARVIS_SILERO_VAD_THRESHOLD` (default `0.5`)
- `JARVIS_VAD_RMS_THRESHOLD` (default `700`)
- `JARVIS_COMMAND_SILENCE_MS` (default `1000`)
- `JARVIS_COMMAND_MAX_SEC` (default `10`)
- `JARVIS_COMMAND_START_TIMEOUT` (default `3`)
- `JARVIS_WHISPER_MODEL` (default `small`)
- `JARVIS_WHISPER_BEAM_SIZE` (default `1`, диапазон `1..3`)
- `JARVIS_AUDIO_DEVICE` (optional: индекс или подстрока имени устройства ввода)
- `JARVIS_TTS_VOLUME` (default `0.8`; можно задавать как `0..1` или `0..100`)


### Wake word models (ONNX)

Для Windows + Python 3.12 в проекте используется только ONNX backend для `openWakeWord`:

- `tflite-runtime` здесь **не используется** (и не требуется).
- Нужно положить ONNX-модели в `models/openwakeword/` (или указать прямой путь через `JARVIS_WAKE_MODEL_PATH`).
- Источник моделей: HuggingFace репозиторий `davidscripka/openwakeword`.

Обязательные файлы:

- `melspectrogram.onnx`
- `embedding_model.onnx`
- `hey_jarvis_v0.1.onnx` (или другой wake-model `.onnx`)

Если модель не найдена, Jarvis корректно отключит Voice mode и покажет инструкцию с нужными файлами.

Опционально можно включить авто-скачивание отсутствующих ONNX файлов:

```bash
set JARVIS_AUTO_DOWNLOAD_MODELS=1
```

По умолчанию авто-скачивание выключено, поэтому Voice mode остается оффлайн без неожиданных сетевых запросов.

### Ограничение для Windows + Python 3.12

- `tflite-runtime` не поддерживается для Windows + Python 3.12 в этом проекте.
- Wake word работает только через ONNX (`openWakeWord` + `onnxruntime`).
- Никакие cloud/internet STT сервисы не используются: распознавание выполняется локально через `faster-whisper`.

### Troubleshooting

- `sounddevice` не видит микрофон: укажи `JARVIS_AUDIO_DEVICE` (например индекс из `sounddevice.query_devices()`) и проверь драйвер/разрешения Windows.
- Wake word срабатывает редко/часто: подстрой `JARVIS_WAKE_THRESHOLD` и/или `JARVIS_WAKE_COOLDOWN`.
- Команда обрезается или не детектится: подстрой `JARVIS_SILERO_VAD_THRESHOLD` (Silero) или `JARVIS_VAD_RMS_THRESHOLD` (lite), а также `JARVIS_COMMAND_SILENCE_MS`, `JARVIS_COMMAND_START_TIMEOUT`.
- На Windows/Python 3.12 `webrtcvad` убран из обязательных зависимостей, чтобы установка проходила без Visual C++ Build Tools.
- Если `torch` слишком тяжёлый или не устанавливается, задай `JARVIS_VAD_BACKEND=lite`.

## Архитектура

- `main.py` — сборка приложения + startup health-check.
- `jarvis_assistant/ollama_client.py` — системный prompt + запрос к `/api/chat`.
- `jarvis_assistant/assistant_core.py` — цикл: input → llm json → execute actions → ответ.
- `jarvis_assistant/executor.py` — набор инструментов управления ПК.
- `jarvis_assistant/ui.py` — Tkinter UI + background-thread без freeze + интеграция voice loop.
- `jarvis_assistant/voice.py` — оффлайн voice pipeline (openWakeWord + VAD + faster-whisper + TTS).
- `jarvis_assistant/memory_store.py` — долговременная память (`facts/preferences`).
- `jarvis_assistant/logger.py` — JSONL-логгер.

## Важно

- Безопасность намеренно минимально ограничена: при сомнениях ассистент должен спрашивать пользователя.
- Любое реальное действие на ПК исполняется только кодом через `actions` из JSON от LLM.
