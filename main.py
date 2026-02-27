from __future__ import annotations

from threading import Event

from jarvis_assistant.assistant_core import JarvisAssistant
from jarvis_assistant.config import CONFIG
from jarvis_assistant.logger import JsonlLogger
from jarvis_assistant.memory_store import MemoryStore
from jarvis_assistant.ollama_client import OllamaClient
from jarvis_assistant.ui import JarvisApp


def main() -> None:
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
