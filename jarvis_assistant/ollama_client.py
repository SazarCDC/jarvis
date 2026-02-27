from __future__ import annotations

import json
import logging
from typing import Any

import requests
from pydantic import ValidationError

from jarvis_assistant.models import LLMDecision


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
}


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
- Основной язык общения — русский.
- Следуй языку пользователя: если пользователь пишет по-английски, можно отвечать на английском.
- Английские термины и короткие фразы допустимы в техническом контексте.
- НЕ используй китайский или другие нерелевантные языки, если пользователь явно не перешёл на них.
- Если есть сомнения по языку, используй русский.
- LLM не исполняет код сама. Только планирует в actions.
- Для chat/question обычно формируй response, actions опционально.
- Для action формируй конкретные шаги в actions.
- Допустимые значения actions[].type: cmd, powershell, launch, search, write_file, read_file, keyboard, mouse, window, screenshot, clipboard, wait, browser.
- НИКОГДА не используй actions[].type вне списка выше. Значения вроде "tool", "open_settings" и любые другие — запрещены.
- Пример открытия блокнота: {"intent":"action","thought":"Открываю блокнот","confidence":0.92,"ask_user":null,"response":"Открываю блокнот.","memory_update":null,"actions":[{"type":"launch","path":"notepad.exe","args":{}}]}
- Для actions[].path ЗАПРЕЩЕНЫ неэкранированные обратные слэши (`\`) в JSON-строках.
- Допустимые форматы пути:
  1) короткое имя exe (пример: `chrome.exe`, `notepad.exe`)
  2) Windows путь с двойными слэшами (пример: `C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe`)
  3) путь с forward slash (пример: `C:/Program Files/Google/Chrome/Application/chrome.exe`)
- Пример открытия Chrome: {"intent":"action","thought":"Открываю Chrome","confidence":0.92,"ask_user":null,"response":"Открываю Chrome.","memory_update":null,"actions":[{"type":"launch","path":"chrome.exe","args":{}}]}
- Пример открытия Блокнота: {"intent":"action","thought":"Открываю блокнот","confidence":0.92,"ask_user":null,"response":"Открываю блокнот.","memory_update":null,"actions":[{"type":"launch","path":"notepad.exe","args":{}}]}
- Пример открытия параметров: {"intent":"action","thought":"Открываю параметры Windows","confidence":0.91,"ask_user":null,"response":"Открываю параметры.","memory_update":null,"actions":[{"type":"launch","path":"ms-settings:","args":{}}]}
- Пример запуска команды: {"intent":"action","thought":"Показываю список файлов","confidence":0.86,"ask_user":null,"response":"Сейчас покажу список файлов.","memory_update":null,"actions":[{"type":"cmd","command":"dir","args":{}}]}
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
        normalized = self._normalize_decision_dict(parsed)
        try:
            return LLMDecision.model_validate(normalized)
        except ValidationError:
            logger.debug(
                "LLM decision validation failed, using safe fallback. raw=%r parsed=%r normalized=%r",
                raw,
                parsed,
                normalized,
            )
            return LLMDecision.model_validate(
                {
                    "intent": "question",
                    "thought": "Нужно уточнение, чтобы выполнить запрос корректно.",
                    "confidence": 0.0,
                    "ask_user": "Уточните, пожалуйста, что именно выполнить.",
                    "response": "Я понял запрос, но мне нужно уточнение.",
                    "memory_update": None,
                    "actions": [],
                }
            )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            if "Invalid \\escape" not in str(exc):
                logger.debug("LLM JSON parsing failed (non-recoverable): %s", exc)
                return OllamaClient._safe_parse_fallback()

        repaired_raw = OllamaClient._escape_invalid_backslashes_in_json_strings(raw)
        try:
            return json.loads(repaired_raw)
        except json.JSONDecodeError as exc:
            logger.debug("LLM JSON parsing failed after repair: %s", exc)
            return OllamaClient._safe_parse_fallback()

    @staticmethod
    def _escape_invalid_backslashes_in_json_strings(raw: str) -> str:
        result: list[str] = []
        in_string = False
        i = 0
        length = len(raw)

        while i < length:
            ch = raw[i]

            if ch == '"':
                backslash_count = 0
                j = i - 1
                while j >= 0 and raw[j] == "\\":
                    backslash_count += 1
                    j -= 1
                if backslash_count % 2 == 0:
                    in_string = not in_string
                result.append(ch)
                i += 1
                continue

            if in_string and ch == "\\":
                next_char = raw[i + 1] if i + 1 < length else ""
                if next_char in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
                    result.append(ch)
                    i += 1
                    continue
                if next_char == "u" and i + 5 < length and all(c in "0123456789abcdefABCDEF" for c in raw[i + 2 : i + 6]):
                    result.append(ch)
                    i += 1
                    continue

                result.append("\\\\")
                i += 1
                continue

            result.append(ch)
            i += 1

        return "".join(result)

    @staticmethod
    def _safe_parse_fallback() -> dict[str, Any]:
        return {
            "intent": "question",
            "thought": "Ошибка формата JSON-ответа LLM, нужно уточнение.",
            "confidence": 0.0,
            "ask_user": "Я понял команду, но возникла ошибка формата ответа. Уточните, пожалуйста.",
            "response": "Я понял команду, но возникла ошибка формата ответа. Уточните, пожалуйста.",
            "memory_update": None,
            "actions": [],
        }

    @staticmethod
    def _normalize_decision_dict(parsed: dict[str, Any]) -> dict[str, Any]:
        normalized = parsed.copy() if isinstance(parsed, dict) else {}
        notes: list[str] = []

        actions = normalized.get("actions")
        if not isinstance(actions, list):
            normalized["actions"] = []
            if actions is not None:
                notes.append("исправил формат действий")
            actions = normalized["actions"]

        cleaned_actions: list[dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                notes.append("убрал некорректное действие")
                continue

            fixed = action.copy()
            action_type = fixed.get("type")

            if action_type == "tool":
                fixed["type"] = "launch"
                notes.append("уточнил тип действия")
            elif action_type == "open_settings":
                fixed["type"] = "launch"
                fixed["path"] = "ms-settings:"
                notes.append("уточнил действие открытия параметров")
            elif not action_type:
                command = str(fixed.get("command") or "").strip()
                path = str(fixed.get("path") or "").strip()
                if command:
                    fixed["type"] = "powershell" if ("Get-" in command or "Set-" in command) else "cmd"
                    notes.append("добавил тип действия")
                elif path:
                    fixed["type"] = "launch"
                    notes.append("добавил тип действия")
                else:
                    notes.append("убрал неполное действие")
                    continue

            if fixed.get("type") not in ALLOWED_ACTION_TYPES:
                notes.append("убрал неподдерживаемое действие")
                continue

            cleaned_actions.append(fixed)

        normalized["actions"] = cleaned_actions

        normalized["response"] = OllamaClient._normalize_language_text(normalized.get("response"))
        normalized["ask_user"] = OllamaClient._normalize_language_text(normalized.get("ask_user"))

        if notes:
            note = "Я немного уточнил план действий."
            if not normalized.get("response"):
                normalized["response"] = note
            if not normalized.get("ask_user") and not cleaned_actions:
                normalized["ask_user"] = "Уточните, пожалуйста, что именно выполнить."
                normalized["intent"] = "question"

        return normalized

    @staticmethod
    def _normalize_language_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None

        # Запрещаем нерелевантные CJK-ответы и заменяем их безопасной фразой на русском.
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return "Я отвечу по-русски. Уточните, пожалуйста, запрос."

        return text
