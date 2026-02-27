from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote_plus

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
- НЕ используй китайский или другие нерелевантные языки.
- metadata.datetime уже передаётся во входе. Если спрашивают время/дату, используй его реальное значение и никогда не выводи шаблоны вроде {datetime}.
- LLM не исполняет код сама. Только планирует actions.
- Допустимые actions[].type: cmd, powershell, launch, search, write_file, read_file, keyboard, mouse, window, screenshot, clipboard, wait, browser.
- search = только поиск файлов/папок на диске.
- browser = интернет/сайты/новости/погода/картинки/поиск в сети.
- Для запросов "в интернете", "в сети", "погода", "новости", "картинки", "фото" всегда используй browser.
- Для browser в response пиши voice-friendly фразы без URL (например: "Открываю результаты в браузере.").
- Для «открой/запусти <приложение>» всегда используй launch (например: notepad.exe, calc.exe, chrome.exe, ms-settings:).
- window используй ТОЛЬКО для управления уже открытым окном: activate/minimize/maximize/close.
- Для window всегда заполняй args.command, иначе действие недопустимо.
- «Открой параметры» = launch с path="ms-settings:" (НЕ window).
- Пример: «Открой блокнот» -> {"type":"launch","path":"notepad.exe","args":{}}.
- Пример: «Закрой блокнот» -> {"type":"window","args":{"command":"close","title":"Блокнот"}}.
- Пример: «Открой параметры» -> {"type":"launch","path":"ms-settings:","args":{}}.
- Если нужно принудительно закрыть блокнот, используй cmd taskkill: {"type":"cmd","command":"taskkill /IM notepad.exe /F","args":{}}.
- Пример "найди котиков в интернете": {"intent":"action","thought":"Открываю поиск картинок","confidence":0.92,"ask_user":null,"response":"Открываю результаты поиска в браузере.","memory_update":null,"actions":[{"type":"browser","command":"https://duckduckgo.com/?q=котики&iax=images&ia=images","args":{}}]}.
- Если неуверен, заполни ask_user.
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
        user_input = self._extract_user_input(context_messages)
        normalized = self._normalize_decision_dict(parsed, user_input=user_input)
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
    def _extract_user_input(context_messages: list[dict[str, str]]) -> str:
        if not context_messages:
            return ""
        raw = context_messages[-1].get("content", "")
        try:
            parsed = json.loads(raw)
            return str(parsed.get("user_input", ""))
        except Exception:
            return ""

    @staticmethod
    def _normalize_decision_dict(parsed: dict[str, Any], user_input: str = "") -> dict[str, Any]:
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

            fixed = OllamaClient._normalize_action_routing(fixed, user_input)
            if not fixed:
                notes.append("убрал некорректное действие")
                continue

            if fixed.get("type") not in ALLOWED_ACTION_TYPES:
                notes.append("убрал неподдерживаемое действие")
                continue

            cleaned_actions.append(fixed)

        normalized["actions"] = cleaned_actions
        normalized["response"] = OllamaClient._normalize_language_text(normalized.get("response"))
        normalized["ask_user"] = OllamaClient._normalize_language_text(normalized.get("ask_user"))

        if notes:
            if not normalized.get("response"):
                normalized["response"] = "Я немного уточнил план действий."
            if not normalized.get("ask_user") and not cleaned_actions:
                if any(token in user_input.lower() for token in ("открой", "запусти")):
                    normalized["ask_user"] = "Что именно открыть?"
                else:
                    normalized["ask_user"] = "Уточните, пожалуйста, что именно выполнить."
                normalized["intent"] = "question"

        return normalized

    @staticmethod
    def _normalize_action_routing(action: dict[str, Any], user_input: str) -> dict[str, Any] | None:
        fixed = action.copy()
        action_type = str(fixed.get("type") or "").lower()
        command = str(fixed.get("command") or "")
        command_lower = command.lower().strip()
        intent_text = user_input.lower()
        is_open_request = any(token in intent_text for token in ("открой", "запусти"))

        if action_type == "window" and is_open_request:
            if "блокнот" in intent_text:
                return {"type": "launch", "path": "notepad.exe", "args": {}}
            if "параметры" in intent_text:
                return {"type": "launch", "path": "ms-settings:", "args": {}}
            if "хром" in intent_text or "chrome" in intent_text:
                return {"type": "launch", "path": "chrome.exe", "args": {}}
            return None

        if action_type == "window":
            args = fixed.get("args") if isinstance(fixed.get("args"), dict) else {}
            window_command = str(args.get("command") or "").strip().lower()
            if not window_command:
                return None

        if action_type == "launch" and command_lower:
            if command_lower.startswith("taskkill"):
                fixed["type"] = "cmd"
            elif command_lower.startswith("powershell") or " get-" in f" {command_lower}" or " set-" in f" {command_lower}":
                fixed["type"] = "powershell"
            elif command_lower.startswith("cmd") or command_lower.startswith("dir") or command_lower.startswith("cd"):
                fixed["type"] = "cmd"

        if action_type == "search" and OllamaClient._is_web_intent(intent_text):
            query = quote_plus(user_input.strip())
            fixed["type"] = "browser"
            fixed["command"] = f"https://duckduckgo.com/?q={query}"

        if fixed.get("type") == "launch":
            target = f"{fixed.get('path') or ''} {fixed.get('command') or ''}".lower()
            if "chrome" in target and (not fixed.get("path") or fixed.get("path") in {"chrome", "google chrome"}):
                fixed["path"] = "chrome.exe"
            if "ms-settings" in target:
                fixed["path"] = "ms-settings:"

        return fixed

    @staticmethod
    def _is_web_intent(text: str) -> bool:
        markers = ["в интернете", "в сети", "на сайте", "погода", "новости", "картинки", "фото"]
        return any(marker in text for marker in markers)

    @staticmethod
    def _normalize_language_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None

        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return "Я отвечу по-русски. Уточните, пожалуйста, запрос."

        return text
