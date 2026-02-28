from __future__ import annotations

import json
import logging
from typing import Any

import requests
from pydantic import ValidationError

from jarvis_assistant.models import LLMDecision, RouterDecision

logger = logging.getLogger(__name__)

ALLOWED_ACTION_TYPES = {
    "cmd",
    "powershell",
    "launch",
    "search",
    "write_file",
    "read_file",
    "keyboard",
    "mouse",
    "window",
    "screenshot",
    "clipboard",
    "wait",
    "browser",
    "audio",
    "web_search",
    "web_fetch",
    "web_extract",
    "monitor",
}

ROUTER_SYSTEM_PROMPT = """
Ты роутер Jarvis. Верни только JSON: {"route":"chat|question|web-research|pc-action|hybrid","thought":"...","confidence":0.0}.
web-research выбирай, когда нужны факты/поиск в интернете; pc-action — действия на ПК; hybrid — и то и другое.
""".strip()

CHAT_SYSTEM_PROMPT = """
Ты Jarvis — дружелюбный и уместно остроумный помощник-друг.
Тон: живой, без токсичности и без клоунады. Если пользователь раздражен — будь прямее и короче.
Верни только JSON: intent/thought/confidence/ask_user/response/memory_update/actions/continue.
Для chat обычно actions=[] и continue=0.
""".strip()

AGENT_SYSTEM_PROMPT = """
Ты агент Jarvis для задач на ПК и в интернете. Верни только JSON:
{
  "intent":"chat|question|action|web|hybrid|noise",
  "thought":"кратко",
  "confidence":0.0,
  "ask_user":null,
  "response":"текст",
  "memory_update":{"facts":{},"preferences":{}},
  "actions":[{"type":"...","command":"...","path":"...","args":{}}],
  "continue":0
}
Правила:
- Если нужны факты из сети, сначала используй web_search -> web_fetch -> web_extract.
- Для действий на ПК используй инструменты честно; если не можешь выполнить — ask_user или response с ограничением.
- Если нужны следующие шаги, ставь continue=1.
- Если задача завершена, continue=0.
- Допустимые actions: cmd,powershell,launch,search,write_file,read_file,keyboard,mouse,window,screenshot,clipboard,wait,browser,audio,web_search,web_fetch,web_extract,monitor.
""".strip()


class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int = 90) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
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
                return False, f"Не удалось подключиться к Ollama ({self.host}): {exc}."
        return False, f"Ollama не отвечает по адресу {self.host}."

    def route(self, metadata: dict[str, Any]) -> RouterDecision:
        parsed = self._chat_json(ROUTER_SYSTEM_PROMPT, metadata, schema=RouterDecision.model_json_schema())
        normalized = self._normalize_router_dict(parsed)
        try:
            return RouterDecision.model_validate(normalized)
        except ValidationError:
            return RouterDecision(route="chat", thought="fallback", confidence=0.3)

    def chat(self, metadata: dict[str, Any]) -> LLMDecision:
        return self._decision(CHAT_SYSTEM_PROMPT, metadata)

    def agent_step(self, metadata: dict[str, Any]) -> LLMDecision:
        return self._decision(AGENT_SYSTEM_PROMPT, metadata)

    def _decision(self, prompt: str, metadata: dict[str, Any]) -> LLMDecision:
        parsed = self._chat_json(prompt, metadata, schema=LLMDecision.model_json_schema(by_alias=True))
        normalized = self._normalize_decision_dict(parsed)
        try:
            return LLMDecision.model_validate(normalized)
        except ValidationError:
            logger.debug("Invalid LLM decision: %r", normalized)
            return LLMDecision.model_validate(
                {
                    "intent": "question",
                    "thought": "fallback",
                    "confidence": 0.2,
                    "ask_user": "Уточните, пожалуйста, что именно нужно сделать.",
                    "response": "Уточните, пожалуйста, что именно нужно сделать.",
                    "memory_update": None,
                    "actions": [],
                    "continue": 0,
                }
            )

    def _chat_json(self, system_prompt: str, metadata: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(metadata, ensure_ascii=False)},
            ],
            "stream": False,
            "format": schema or "json",
            "options": {"temperature": 0},
        }
        response = self.session.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        raw = response.json().get("message", {}).get("content", "{}")
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw_text: str) -> dict[str, Any]:
        text = (raw_text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
        return {}

    @staticmethod
    def _normalize_decision_dict(parsed: dict[str, Any]) -> dict[str, Any]:
        data = parsed if isinstance(parsed, dict) else {}
        actions = data.get("actions") if isinstance(data.get("actions"), list) else []
        cleaned = [a for a in actions if isinstance(a, dict) and a.get("type") in ALLOWED_ACTION_TYPES]

        raw_confidence = data.get("confidence", 0.7)
        try:
            confidence = float(raw_confidence)
        except Exception:
            confidence = 0.7
        confidence = max(0.0, min(1.0, confidence))

        raw_continue = data.get("continue", 0)
        try:
            continue_value = int(raw_continue)
        except Exception:
            continue_value = 0
        continue_value = 1 if continue_value > 0 else 0

        intent = str(data.get("intent", "question"))
        if intent not in {"chat", "question", "action", "noise", "web", "hybrid"}:
            intent = "question"

        return {
            "intent": intent,
            "thought": str(data.get("thought", "")),
            "confidence": confidence,
            "ask_user": data.get("ask_user"),
            "response": data.get("response"),
            "memory_update": data.get("memory_update"),
            "actions": cleaned,
            "continue": continue_value,
        }

    @staticmethod
    def _normalize_router_dict(parsed: dict[str, Any]) -> dict[str, Any]:
        data = parsed if isinstance(parsed, dict) else {}
        route = str(data.get("route", "chat"))
        if route not in {"chat", "question", "web-research", "pc-action", "hybrid"}:
            route = "chat"

        raw_confidence = data.get("confidence", 0.8)
        try:
            confidence = float(raw_confidence)
        except Exception:
            confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))

        return {
            "route": route,
            "thought": str(data.get("thought", "")),
            "confidence": confidence,
        }
