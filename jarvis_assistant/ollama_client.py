from __future__ import annotations

import json
from typing import Any

import requests

from jarvis_assistant.models import LLMDecision


SYSTEM_PROMPT = """
Ты управляющий интеллект ассистента Jarvis.
Всегда отвечай ТОЛЬКО валидным JSON-объектом без markdown.
Схема:
{
  "intent": "chat|question|action|noise",
  "thought": "короткий план",
  "confidence": 0.0,
  "ask_user": null,
  "response": "сообщение пользователю",
  "memory_update": {"facts": {...}, "preferences": {...}},
  "actions": [{"type": "...", "command": "...", "path": "...", "args": {...}}]
}
Правила:
- LLM не исполняет код сама. Только планирует в actions.
- Для chat/question обычно формируй response, actions опционально.
- Для action формируй конкретные шаги tools.
- Если неуверен, заполни ask_user.
- Для noise дай человеческий response с просьбой уточнить.
- Если используешь поиск/интернет, укажи это в response.
""".strip()


class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int = 90) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

        # КЛЮЧЕВО: не брать прокси из окружения (Hiddify / системный proxy)
        self.session = requests.Session()
        self.session.trust_env = False

    def health_check(self) -> tuple[bool, str]:
        try:
            self.session.head(f"{self.host}/", timeout=3)
            return True, "ok"
        except Exception:
            try:
                r = self.session.get(f"{self.host}/api/tags", timeout=5)
                if r.ok:
                    return True, "ok"
            except Exception as exc:
                return False, f"Не удалось подключиться к Ollama ({self.host}): {exc}. Запусти 'ollama serve'."
        return False, f"Ollama не отвечает по адресу {self.host}. Запусти 'ollama serve'."

    def plan(self, context_messages: list[dict[str, str]]) -> LLMDecision:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *context_messages]
        payload = {"model": self.model, "messages": messages, "stream": False}

        response = self.session.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        raw = response.json().get("message", {}).get("content", "{}")
        parsed = self._parse_json(raw)
        return LLMDecision.model_validate(parsed)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()
        return json.loads(raw)
