from __future__ import annotations

import json
import re
from datetime import datetime
from threading import Event
from typing import Any, Callable
from urllib.parse import quote_plus

from jarvis_assistant.executor import ActionExecutor
from jarvis_assistant.logger import JsonlLogger
from jarvis_assistant.memory_store import MemoryStore
from jarvis_assistant.models import ActionResult, ActionSpec, LLMDecision
from jarvis_assistant.ollama_client import OllamaClient


class JarvisAssistant:
    def __init__(
        self,
        llm: OllamaClient,
        memory: MemoryStore,
        logger: JsonlLogger,
        stop_event: Event,
        max_history_messages: int,
        command_timeout_sec: int,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.logger = logger
        self.stop_event = stop_event
        self.executor = ActionExecutor(stop_event=stop_event, command_timeout_sec=command_timeout_sec)
        self.max_history_messages = max_history_messages
        self.history: list[dict[str, str]] = []
        self.pending: dict[str, Any] | None = None

    def stop(self) -> None:
        self.stop_event.set()

    def process_user_message(self, user_text: str, status_cb: Callable[[str], None]) -> str:
        self.stop_event.clear()
        self.history.append({"role": "user", "content": user_text})
        self.logger.log("input", {"text": user_text})

        pending_reply = self._handle_pending_selection(user_text, status_cb)
        if pending_reply:
            self.history.append({"role": "assistant", "content": pending_reply})
            return pending_reply

        deterministic_reply = self._deterministic_time_date_reply(user_text)
        if deterministic_reply:
            self.pending = None
            self.history.append({"role": "assistant", "content": deterministic_reply})
            status_cb("Idle")
            return deterministic_reply

        shortcut_reply = self._deterministic_web_shortcut(user_text)
        if shortcut_reply:
            status_cb("Acting")
            action = shortcut_reply["action"]
            result = self.executor.run_actions([action])[0]
            reply = shortcut_reply["reply"]
            self.history.append({"role": "assistant", "content": reply})
            self.logger.log("shortcut_action", {"action": action.model_dump(), "result": result.model_dump()})
            status_cb("Idle")
            return reply

        context = self._build_context(user_text)
        status_cb("Thinking")

        decision = self.llm.plan(context)
        self.logger.log("llm_json_output", decision.model_dump())

        if self.stop_event.is_set():
            return "Остановлено"

        memory_update = self._extract_memory_update(decision)
        if memory_update:
            self.memory.apply_updates(memory_update)
            self.logger.log("memory_update", memory_update)

        if self._needs_clarification(user_text, decision):
            reply = "Не до конца понял команду. Уточните, пожалуйста, что именно открыть или найти."
            self.history.append({"role": "assistant", "content": reply})
            status_cb("Idle")
            return reply

        if decision.ask_user:
            reply = decision.ask_user
            self.history.append({"role": "assistant", "content": reply})
            status_cb("Idle")
            return reply

        action_results: list[ActionResult] = []
        if decision.actions:
            status_cb("Acting")
            action_results = self.executor.run_actions(decision.actions)
            self.logger.log("actions_executed", [a.model_dump() for a in decision.actions])
            self.logger.log("action_results", [r.model_dump() for r in action_results])
            self._set_pending_from_results(action_results)

        recovery_reply = self._build_recovery_reply(user_text, decision.actions, action_results)
        if recovery_reply:
            reply = recovery_reply
        else:
            reply = self._compose_reply(decision, action_results)

        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-self.max_history_messages :]
        status_cb("Idle")
        return reply

    def _build_context(self, user_text: str) -> list[dict[str, str]]:
        memory = self.memory.get_memory_payload()
        metadata = {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "memory": memory,
            "conversation_tail": self.history[-self.max_history_messages :],
            "user_input": user_text,
        }
        return [{"role": "user", "content": json.dumps(metadata, ensure_ascii=False)}]

    @staticmethod
    def _extract_memory_update(decision: LLMDecision) -> dict:
        update = decision.memory_update
        return update if isinstance(update, dict) else {}

    @staticmethod
    def _looks_actionable(user_text: str) -> bool:
        text = user_text.lower()
        hints = ["открой", "запусти", "закрой", "найди", "покажи", "сделай", "удали", "создай"]
        return any(hint in text for hint in hints)

    def _needs_clarification(self, user_text: str, decision: LLMDecision) -> bool:
        return self._looks_actionable(user_text) and not decision.actions and not decision.ask_user

    def _deterministic_time_date_reply(self, user_text: str) -> str | None:
        text = user_text.lower()
        now = datetime.now()

        if any(phrase in text for phrase in ["сколько время", "сколько времени", "который час", "время сейчас", "текущее время"]):
            return f"Сейчас {now.strftime('%H:%M')}."

        if any(phrase in text for phrase in ["какая дата", "какое сегодня число", "сегодняшняя дата", "какой сегодня день", "день недели"]):
            weekdays = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
            weekday = weekdays[now.weekday()]
            return f"Сегодня {now.strftime('%d.%m.%Y')}, {weekday}."

        return None

    def _deterministic_web_shortcut(self, user_text: str) -> dict[str, Any] | None:
        text = user_text.lower().strip()
        if "погода" in text:
            query = quote_plus(user_text.strip())
            action = ActionSpec(type="browser", command=f"https://duckduckgo.com/?q={query}", args={})
            return {
                "action": action,
                "reply": "Открыл прогноз погоды в браузере. Хотите текущую температуру или прогноз на неделю?",
            }

        if ("в интернете" in text or "в сети" in text or "на сайте" in text) and "найд" in text:
            query = quote_plus(user_text.strip())
            images = any(word in text for word in ["картинки", "картинку", "фото", "изображения", "котиков"])
            if images:
                url = f"https://duckduckgo.com/?q={query}&iax=images&ia=images"
            else:
                url = f"https://duckduckgo.com/?q={query}"
            action = ActionSpec(type="browser", command=url, args={})
            return {"action": action, "reply": "Открываю результаты поиска в браузере."}

        return None

    def _handle_pending_selection(self, user_text: str, status_cb: Callable[[str], None]) -> str | None:
        if not self.pending:
            return None

        index = self._parse_choice_index(user_text)
        if index is None:
            choice_text = user_text.strip().lower()
            if self.pending.get("type") == "choose_recovery_option" and choice_text in {"да", "yes"}:
                index = 0
            elif self.pending.get("type") == "choose_recovery_option" and choice_text in {"нет", "no", "0"}:
                self.pending = None
                return "Хорошо, отменяю действие."
            else:
                return None

        items = self.pending.get("items", [])
        if not isinstance(items, list) or not (0 <= index < len(items)):
            return "Выберите номер из предложенных вариантов."

        selected = items[index]
        action_payload = selected.get("action")
        if not isinstance(action_payload, dict):
            self.pending = None
            return "Не удалось обработать выбор. Попробуйте снова."

        action = ActionSpec.model_validate(action_payload)
        status_cb("Acting")
        result = self.executor.run_actions([action])[0]
        self.logger.log("pending_action_selected", {"selected": index + 1, "action": action_payload, "result": result.model_dump()})

        preference = selected.get("preference")
        if isinstance(preference, dict):
            path = preference.get("path")
            value = preference.get("value")
            if isinstance(path, list) and path:
                self.memory.set_preference(path, value)

        self.pending = None
        status_cb("Idle")
        return self._compose_reply(LLMDecision(intent="action", thought="", confidence=1.0, response="Готово.", actions=[]), [result])

    @staticmethod
    def _parse_choice_index(user_text: str) -> int | None:
        text = user_text.strip().lower()
        if re.fullmatch(r"\d+", text):
            return int(text) - 1
        mapping = {"первый": 0, "первая": 0, "второй": 1, "вторая": 1, "третий": 2, "третья": 2}
        return mapping.get(text)

    def _set_pending_from_results(self, action_results: list[ActionResult]) -> None:
        for result in action_results:
            if result.ok and result.action_type == "search" and result.paths:
                items = []
                for path in result.paths[:5]:
                    items.append({
                        "label": path,
                        "action": {"type": "launch", "path": path, "args": {}},
                    })
                self.pending = {
                    "type": "choose_path",
                    "items": items,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                break

    def _build_recovery_reply(self, user_text: str, actions: list[ActionSpec], action_results: list[ActionResult]) -> str | None:
        failed = [res for res in action_results if not res.ok]
        if not failed:
            return None

        first_fail = failed[0]
        error_text = first_fail.error_message or "действие завершилось с ошибкой"
        user_lower = user_text.lower()

        if self._is_notepad_close_case(user_lower, actions):
            options = [
                {
                    "label": "Закрыть принудительно через taskkill",
                    "action": {"type": "cmd", "command": "taskkill /IM notepad.exe /F", "args": {}},
                    "preference": {"path": ["app_close_method", "notepad"], "value": "taskkill"},
                }
            ]
            self.pending = {"type": "choose_recovery_option", "items": options, "created_at": datetime.now().isoformat(timespec="seconds")}
            return (
                f"Не получилось закрыть Блокнот: {error_text}. "
                "Могу закрыть его принудительно через taskkill. Выполнить? (да/нет)"
            )

        if self._is_chrome_open_case(user_lower, actions, failed):
            options = [
                {
                    "label": "Открыть Edge",
                    "action": {"type": "launch", "path": "msedge.exe", "args": {}},
                    "preference": {"path": ["default_browser"], "value": "edge"},
                },
                {
                    "label": "Открыть браузер по умолчанию",
                    "action": {"type": "browser", "command": "https://google.com", "args": {}},
                    "preference": {"path": ["default_browser"], "value": "system"},
                },
            ]
            self.pending = {"type": "choose_recovery_option", "items": options, "created_at": datetime.now().isoformat(timespec="seconds")}
            return (
                f"Не получилось открыть Chrome: {error_text}. "
                "Выберите вариант: 1) Edge, 2) браузер по умолчанию."
            )

        if self._is_window_command_recovery_case(first_fail):
            window_title = self._infer_window_title(user_lower)
            options = [
                {"label": "Активировать окно", "action": {"type": "window", "args": {"command": "activate", "title": window_title}}},
                {"label": "Закрыть окно", "action": {"type": "window", "args": {"command": "close", "title": window_title}}},
                {"label": "Свернуть окно", "action": {"type": "window", "args": {"command": "minimize", "title": window_title}}},
                {"label": "Развернуть окно", "action": {"type": "window", "args": {"command": "maximize", "title": window_title}}},
            ]
            self.pending = {"type": "choose_recovery_option", "items": options, "created_at": datetime.now().isoformat(timespec="seconds")}
            return (
                "Уточните, что сделать с окном: 1) активировать 2) закрыть 3) свернуть 4) развернуть 0) отмена."
            )

        return None

    @staticmethod
    def _is_notepad_close_case(user_text: str, actions: list[ActionSpec]) -> bool:
        if "закрой" in user_text and "блокнот" in user_text:
            return True
        for action in actions:
            title = str(action.args.get("title", "")).lower()
            cmd = str(action.command or "").lower()
            if action.type == "window" and action.args.get("command") == "close" and ("блокнот" in title or "notepad" in title):
                return True
            if "taskkill" in cmd and "notepad" in cmd:
                return True
        return False

    @staticmethod
    def _infer_window_title(user_text: str) -> str:
        if "блокнот" in user_text:
            return "Блокнот"
        if "chrome" in user_text or "хром" in user_text:
            return "Chrome"
        return ""

    @staticmethod
    def _is_window_command_recovery_case(result: ActionResult) -> bool:
        if result.action_type != "window":
            return False
        error_text = (result.error_message or "").lower()
        return "не указана операция для окна" in error_text or "нужен title или process" in error_text

    @staticmethod
    def _is_chrome_open_case(user_text: str, actions: list[ActionSpec], failed: list[ActionResult]) -> bool:
        if "chrome" in user_text and ("открой" in user_text or "запусти" in user_text):
            return True
        failed_not_found = any((item.data or {}).get("error_code") == "FILE_NOT_FOUND" for item in failed)
        for action in actions:
            target = f"{action.path or ''} {action.command or ''}".lower()
            if "chrome" in target and failed_not_found:
                return True
        return False

    @staticmethod
    def _compose_reply(decision: LLMDecision, action_results: list[ActionResult]) -> str:
        base = decision.response or f"План: {decision.thought}"
        if not action_results:
            return base

        lines: list[str] = []
        for result in action_results:
            lines.append(JarvisAssistant._format_result_line(result))

        compact_lines = [line for line in lines if line]
        if not compact_lines:
            return base
        return base + "\n\n" + "\n".join(compact_lines)

    @staticmethod
    def _format_result_line(result: ActionResult) -> str:
        if not result.ok:
            details = result.error_message or "не удалось выполнить действие"
            return f"⚠️ {details}."

        if result.action_type == "search":
            items = result.paths[:5]
            prefix = f"Найдено {len(result.paths)} совпадений."
            if not items:
                return "Совпадений не найдено."
            joined = "; ".join(items)
            return f"{prefix} Примеры: {joined}. Можете сказать номер 1-5, чтобы открыть."

        if result.action_type == "browser":
            return "Открыл в браузере."

        if result.action_type == "screenshot":
            screenshot_path = result.screenshot_path or (result.files[0] if result.files else "")
            return f"Скриншот сохранён: {screenshot_path}."

        if result.action_type == "read_file":
            content = str((result.data or {}).get("content", "")).strip()
            snippet = content[:360]
            if len(content) > 360:
                snippet += "…"
            if snippet:
                return f"Прочитал файл: {snippet} Сказать 'показать дальше' — продолжу."
            return "Файл прочитан, но содержимое пустое."

        if result.action_type == "write_file" and result.files:
            return f"Файл сохранён: {result.files[0]}."

        if result.action_type == "launch" and result.paths:
            return "Запустил приложение."

        if result.stdout:
            return result.stdout.strip()[:300]

        if result.paths:
            return f"Обработано путей: {len(result.paths)}."

        if result.files:
            return f"Готово. Файлы: {', '.join(result.files[:3])}."

        return f"Действие {result.action_type} выполнено."
