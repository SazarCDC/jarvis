from __future__ import annotations

import json
from datetime import datetime
from threading import Event
from typing import Callable

from jarvis_assistant.executor import ActionExecutor
from jarvis_assistant.logger import JsonlLogger
from jarvis_assistant.memory_store import MemoryStore
from jarvis_assistant.models import ActionResult, LLMDecision
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

    def stop(self) -> None:
        self.stop_event.set()

    def process_user_message(self, user_text: str, status_cb: Callable[[str], None]) -> str:
        self.stop_event.clear()
        self.history.append({"role": "user", "content": user_text})
        context = self._build_context(user_text)
        self.logger.log("input", {"text": user_text})
        status_cb("Thinking")

        decision = self.llm.plan(context)
        self.logger.log("llm_json_output", decision.model_dump())

        if self.stop_event.is_set():
            return "Остановлено"

        memory_update = self._extract_memory_update(decision)
        if memory_update:
            self.memory.apply_updates(memory_update)
            self.logger.log("memory_update", memory_update)

        if decision.ask_user:
            reply = decision.ask_user
            self.history.append({"role": "assistant", "content": reply})
            return reply

        action_results: list[ActionResult] = []
        if decision.actions:
            status_cb("Acting")
            action_results = self.executor.run_actions(decision.actions)
            self.logger.log("actions_executed", [a.model_dump() for a in decision.actions])
            self.logger.log("action_results", [r.model_dump() for r in action_results])

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
    def _compose_reply(decision: LLMDecision, action_results: list[ActionResult]) -> str:
        if decision.response:
            base = decision.response
        else:
            base = f"План: {decision.thought}"
        if action_results:
            tail = []
            for idx, result in enumerate(action_results, start=1):
                status = "OK" if result.ok else "FAIL"
                details = result.error_message or result.stdout or ""
                tail.append(f"{idx}. [{status}] {result.action_type}: {details}".strip())
            return base + "\n\nРезультаты действий:\n" + "\n".join(tail)
        return base
