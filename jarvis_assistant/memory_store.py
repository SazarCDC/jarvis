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
        preferences = self._normalized_preferences(self._load(self.preferences_file))
        self._save(self.preferences_file, preferences)
        return {
            "facts": self._load(self.facts_file),
            "preferences": preferences,
        }

    def apply_updates(self, updates: dict[str, Any]) -> None:
        if not updates:
            return
        if "facts" in updates and isinstance(updates["facts"], dict):
            facts = self._load(self.facts_file)
            facts.update(updates["facts"])
            self._save(self.facts_file, facts)
        if "preferences" in updates and isinstance(updates["preferences"], dict):
            prefs = self._normalized_preferences(self._load(self.preferences_file))
            prefs = self._deep_merge_dicts(prefs, updates["preferences"])
            prefs = self._normalized_preferences(prefs)
            self._save(self.preferences_file, prefs)

    def set_preference(self, path: list[str], value: Any) -> None:
        if not path:
            return
        prefs = self._normalized_preferences(self._load(self.preferences_file))
        cursor: dict[str, Any] = prefs
        for key in path[:-1]:
            existing = cursor.get(key)
            if not isinstance(existing, dict):
                existing = {}
                cursor[key] = existing
            cursor = existing
        cursor[path[-1]] = value
        self._save(self.preferences_file, self._normalized_preferences(prefs))

    @staticmethod
    def _deep_merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = MemoryStore._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _normalized_preferences(preferences: dict[str, Any]) -> dict[str, Any]:
        normalized = preferences if isinstance(preferences, dict) else {}
        app_close_method = normalized.get("app_close_method")
        if not isinstance(app_close_method, dict):
            normalized["app_close_method"] = {}
        normalized.setdefault("default_browser", "system")
        return normalized
