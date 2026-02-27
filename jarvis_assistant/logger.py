from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        item = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event_type,
            "payload": payload,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
