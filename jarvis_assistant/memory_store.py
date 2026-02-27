from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.facts_file = self.root / "facts.json"
        self.preferences_file = self.root / "preferences.json"
        self._ensure_file(self.facts_file)
        self._ensure_file(self.preferences_file)

    @staticmethod
    def _ensure_file(path: Path) -> None:
        if not path.exists():
            path.write_text("{}", encoding="utf-8")

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _save(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_memory_payload(self) -> dict[str, Any]:
        return {
            "facts": self._load(self.facts_file),
            "preferences": self._load(self.preferences_file),
        }

    def apply_updates(self, updates: dict[str, Any]) -> None:
        if not updates:
            return
        if "facts" in updates and isinstance(updates["facts"], dict):
            facts = self._load(self.facts_file)
            facts.update(updates["facts"])
            self._save(self.facts_file, facts)
        if "preferences" in updates and isinstance(updates["preferences"], dict):
            prefs = self._load(self.preferences_file)
            prefs.update(updates["preferences"])
            self._save(self.preferences_file, prefs)
