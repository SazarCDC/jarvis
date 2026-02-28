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
        return {"facts": self._load(self.facts_file), "preferences": preferences}

    def apply_updates(self, updates: dict[str, Any]) -> None:
        if not updates:
            return
        if "facts" in updates and isinstance(updates["facts"], dict):
            facts = self._load(self.facts_file)
            facts.update(updates["facts"])
            self._save(self.facts_file, facts)
        if "preferences" in updates and isinstance(updates["preferences"], dict):
            prefs = self._deep_merge_dicts(self._normalized_preferences(self._load(self.preferences_file)), updates["preferences"])
            self._save(self.preferences_file, self._normalized_preferences(prefs))

    def set_preference(self, path: list[str], value: Any) -> None:
        if not path:
            return
        prefs = self._normalized_preferences(self._load(self.preferences_file))
        cursor: dict[str, Any] = prefs
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {}) if isinstance(cursor.get(key, {}), dict) else {}
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
    def _clamp01(value: Any, default: float) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except Exception:
            return default

    @staticmethod
    def _normalized_preferences(preferences: dict[str, Any]) -> dict[str, Any]:
        normalized = preferences if isinstance(preferences, dict) else {}
        normalized.setdefault("app_close_method", {})
        normalized.setdefault("default_browser", "system")
        voice = normalized.setdefault("voice", {})
        voice.setdefault("tts_volume_percent", 80)
        persona = normalized.setdefault("persona", {})
        persona["friendliness"] = MemoryStore._clamp01(persona.get("friendliness", 0.85), 0.85)
        persona["humor"] = MemoryStore._clamp01(persona.get("humor", 0.35), 0.35)
        persona["directness"] = MemoryStore._clamp01(persona.get("directness", 0.55), 0.55)
        dialogue = normalized.setdefault("dialogue", {})
        dialogue.setdefault("style", "friendly")
        return normalized
