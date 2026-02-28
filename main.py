from __future__ import annotations

import subprocess
import time
from threading import Event

import requests

from jarvis_assistant.assistant_core import JarvisAssistant
from jarvis_assistant.config import CONFIG
from jarvis_assistant.logger import JsonlLogger
from jarvis_assistant.memory_store import MemoryStore
from jarvis_assistant.ollama_client import OllamaClient
from jarvis_assistant.ui import JarvisApp



def ensure_ollama_running(host: str) -> None:
    host = host.rstrip("/")

    # ВАЖНО: игнорируем прокси/Hiddify для localhost
    s = requests.Session()
    s.trust_env = False

    def healthy() -> bool:
        try:
            r = s.get(f"{host}/api/tags", timeout=1.5)
            return r.ok
        except Exception:
            return False

    # Уже работает?
    if healthy():
        return

    # Пробуем запустить
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Не найден 'ollama' в PATH. Проверь, что Ollama установлена и доступна из терминала.") from exc
    except Exception as exc:
        raise RuntimeError(f"Не удалось запустить 'ollama serve': {exc}") from exc

    # Ждём дольше (иногда поднимается медленно)
    for _ in range(60):  # до ~30 секунд
        if healthy():
            return
        time.sleep(0.5)

    raise RuntimeError(
        f"Не удалось подключиться к Ollama ({host}). "
        "Проверь: Ollama установлена, порт 11434 слушается, и Hiddify/VPN не ломает localhost."
    )

def main() -> None:
    # Auto-start Ollama if needed
    ensure_ollama_running(CONFIG.ollama_host)

    llm = OllamaClient(host=CONFIG.ollama_host, model=CONFIG.ollama_model)
    ok, message = llm.health_check()
    if not ok:
        raise RuntimeError(message)

    memory = MemoryStore(CONFIG.paths.memory_dir)
    logger = JsonlLogger(CONFIG.paths.log_dir / "events.jsonl")
    assistant = JarvisAssistant(
        llm=llm,
        memory=memory,
        logger=logger,
        stop_event=Event(),
        max_history_messages=CONFIG.agent.max_history,
        command_timeout_sec=CONFIG.runtime.command_timeout_sec,
    )

    app = JarvisApp(assistant=assistant, logger=logger)
    app.mainloop()


if __name__ == "__main__":
    main()
