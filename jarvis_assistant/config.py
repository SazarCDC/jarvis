from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Config:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    logs_dir: Path = Path(os.getenv("JARVIS_LOG_DIR", "logs"))
    memory_dir: Path = Path(os.getenv("JARVIS_MEMORY_DIR", "jarvis_assistant/memory"))
    max_history_messages: int = int(os.getenv("JARVIS_MAX_HISTORY", "20"))
    command_timeout_sec: int = int(os.getenv("JARVIS_COMMAND_TIMEOUT", "120"))


CONFIG = Config()
