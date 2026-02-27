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
    """
    Ensure Ollama is reachable; if not, try to start `ollama serve` automatically.
    Works on Windows; does nothing if Ollama is already running.
    """
    def healthy() -> bool:
        try:
            # /api/tags is what your health_check uses too
            r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=1.5)
            return r.ok
        except Exception:
            return False

    if healthy():
        return

    # Try to start Ollama in background (no console window on Windows)
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Ollama не установлена или не найдена в PATH. Установи Ollama и перезапусти программу."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Не удалось запустить Ollama автоматически: {exc}") from exc

    # Wait for Ollama to come up
    for _ in range(20):  # ~10 seconds max
        if healthy():
            return
        time.sleep(0.5)

    raise RuntimeError(
        f"Не удалось подключиться к Ollama ({host}). Проверь, что Ollama установлена и не блокируется фаерволом."
    )


def main() -> None:
    # Auto-start Ollama if needed
    ensure_ollama_running(CONFIG.ollama_host)

    llm = OllamaClient(host=CONFIG.ollama_host, model=CONFIG.ollama_model)
    ok, message = llm.health_check()
    if not ok:
        raise RuntimeError(message)

    memory = MemoryStore(CONFIG.memory_dir)
    logger = JsonlLogger(CONFIG.logs_dir / "events.jsonl")
    assistant = JarvisAssistant(
        llm=llm,
        memory=memory,
        logger=logger,
        stop_event=Event(),
        max_history_messages=CONFIG.max_history_messages,
        command_timeout_sec=CONFIG.command_timeout_sec,
    )

    app = JarvisApp(assistant)
    app.mainloop()


if __name__ == "__main__":
    main()
