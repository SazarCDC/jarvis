# Jarvis (Ollama + Windows Desktop)

Персональный AI-ассистент с архитектурой **LLM = мозг / код = исполнитель**.

## Возможности

- Диалоговый интерфейс в одном окне (история, ввод, Execute/Send, STOP, статусы `Idle/Listening/Heard wake word/Thinking/Acting/Speaking`).
- Полностью оффлайн voice mode: wake word через `Picovoice Porcupine` (кастомный `.ppn` для «джарвис»), STT команд через `faster-whisper`, TTS через `Piper`.
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
- Wake word в UI/логах: `джарвис` (детекция через `Picovoice Porcupine` + кастомный `.ppn`).
- После wake word ассистент подаёт короткий beep (earcon) вместо фразы «Слушаю», затем записывает команду с микрофона (до тишины или max 10s) и отправляет её в стандартный текстовый pipeline. Это уменьшает самоподхват TTS в STT.
- При включении Voice режима выполняется короткий TTS self-test: «Голосовой режим включен».
- Ответ появляется в чате и озвучивается через `Piper` (dedicated TTS-thread + очередь сообщений, без блокировки UI).
- Piper запускается только в режиме RAW STREAM: текст передаётся в `stdin`, PCM 16-bit (`--output_raw`) читается из `stdout` и воспроизводится через `sounddevice`.
- Если ассистент уже выполняет задачу, на wake word он отвечает «Подожди секунду».
- Кнопка `STOP` останавливает текущий pipeline, voice-listening, текущую запись/транскрипцию и мгновенно прерывает текущую озвучку (`sounddevice.stop()` + очистка очереди TTS).
- В UI есть слайдер `TTS Volume` (0..100), который применяется сразу.

### Voice ENV

- `JARVIS_PICOVOICE_ACCESS_KEY` (обязательно)
- `JARVIS_PICOVOICE_MODEL_PATH` (обязательно: путь к кастомному `.ppn` для «джарвис»)
- `JARVIS_AUDIO_DEVICE_INDEX` (optional: индекс устройства ввода для `PvRecorder`)
- `JARVIS_PORCUPINE_SENSITIVITY` (default `0.7`, диапазон `0.0..1.0`; выше = чувствительнее, но больше ложных срабатываний)
- `JARVIS_WAKE_DEBUG` (default `0`; `1` = лог каждые ~2с с device/rms + подробные wake-события)
- `JARVIS_WAKE_COOLDOWN` (default `1.0`; игнор повторных wake в течение cooldown)
- `JARVIS_WAKE_EARCON` (default `1`; `0` = отключить beep после wake)
- `JARVIS_WAKE_EARCON_FREQ` (default `880.0`; частота beep в Гц)
- `JARVIS_WAKE_EARCON_MS` (default `70`, clamp `20..250`; длительность beep)
- `JARVIS_WAKE_EARCON_GAIN` (default `0.25`, clamp `0..1`; громкость beep)
- `JARVIS_WAKE_POST_TTS_SILENCE_MS` (default `120`, clamp `0..800`; пауза тишины после beep/TTS перед записью)
- `JARVIS_COMMAND_PRE_ROLL_MS` (default `350`; после wake отбрасывает хвост wake/earcon перед записью)
- `JARVIS_COMMAND_START_TIMEOUT` (default `3.5`; ожидание начала речи после wake)
- `JARVIS_VAD_RMS_THRESHOLD` (default `350`)
- `JARVIS_COMMAND_SILENCE_MS` (default `1200`)
- `JARVIS_COMMAND_MAX_SEC` (default `10`)
- `JARVIS_WHISPER_MODEL` (default `small`)
- `JARVIS_WHISPER_BEAM_SIZE` (default `1`, диапазон `1..3`)
- `JARVIS_TTS_BACKEND` (default `piper`)
- `JARVIS_PIPER_MODEL_PATH` (обязательно: путь к `*.onnx` модели Piper)
- `JARVIS_PIPER_EXE_PATH` (обязательно: полный путь к `piper.exe`, TTS работает только через RAW STREAM subprocess)
- `JARVIS_TTS_VOLUME` (default `80`; можно задавать как `0..1` или `0..100`)
- `JARVIS_TTS_RATE` (optional, default `1.0`)

### Troubleshooting

- `PvRecorder` не видит микрофон: укажи `JARVIS_AUDIO_DEVICE_INDEX` и проверь драйвер/разрешения Windows.
- Wake word не срабатывает: проверь `JARVIS_PICOVOICE_ACCESS_KEY`, путь `JARVIS_PICOVOICE_MODEL_PATH` и качество микрофона.
- Команда обрезается или не детектится: подстрой `JARVIS_VAD_RMS_THRESHOLD` и `JARVIS_COMMAND_SILENCE_MS`.
- Voice pipeline полностью оффлайн после получения `Picovoice AccessKey`.
- Нет звука в TTS: проверь `TTS Volume` в UI, `Default Output Device` в Параметрах звука Windows и что `JARVIS_PIPER_MODEL_PATH` указывает на существующую `.onnx` модель.

### Wake word (Porcupine)

1. Получить AccessKey на https://console.picovoice.ai
2. Скачать кастомную wake-model (`.ppn`) для «джарвис»
3. Установить ENV:
   - `JARVIS_PICOVOICE_ACCESS_KEY`
   - `JARVIS_PICOVOICE_MODEL_PATH`
4. Запустить:

```bash
python main.py
```

Чувствительность wake word можно повысить, например:

```bash
setx JARVIS_PORCUPINE_SENSITIVITY "0.85"
```

⚠️ Слишком высокие значения чувствительности могут увеличить количество ложных срабатываний.

### Быстрая настройка Piper (Windows)

1. Скачай оффлайн voice model Piper (`.onnx`) из официального списка голосов Piper (например, ru_RU voice) и положи в локальную папку, например `C:\jarvis\models\piper`.
2. Установи переменные окружения:

```bash
setx JARVIS_PIPER_MODEL_PATH "C:\jarvis\models\piper\ru_RU-....onnx"
setx JARVIS_PIPER_EXE_PATH "C:\jarvis\tools\piper\piper.exe"
```

3. Запусти приложение:

```bash
python main.py
```

## Архитектура

- `main.py` — сборка приложения + startup health-check.
- `jarvis_assistant/ollama_client.py` — системный prompt + запрос к `/api/chat`.
- `jarvis_assistant/assistant_core.py` — цикл: input → llm json → execute actions → ответ.
- `jarvis_assistant/executor.py` — набор инструментов управления ПК.
- `jarvis_assistant/ui.py` — Tkinter UI + background-thread без freeze + интеграция voice loop.
- `jarvis_assistant/voice.py` — оффлайн voice pipeline (Porcupine `.ppn` + PvRecorder + faster-whisper + Piper TTS).
- `jarvis_assistant/memory_store.py` — долговременная память (`facts/preferences`).
- `jarvis_assistant/logger.py` — JSONL-логгер.

## Важно

- Безопасность намеренно минимально ограничена: при сомнениях ассистент должен спрашивать пользователя.
- Любое реальное действие на ПК исполняется только кодом через `actions` из JSON от LLM.
