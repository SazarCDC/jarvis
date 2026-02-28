from __future__ import annotations

from datetime import datetime
from threading import Event
from typing import Any, Callable

from jarvis_assistant.config import CONFIG
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
        self.history_summary: str = ""
        self.pending_task_state: dict[str, Any] | None = None
        self.awaiting_user: bool = False

    def stop(self) -> None:
        self.stop_event.set()

    def process_user_message(self, user_text: str, status_cb: Callable[[str], None]) -> str:
        self.stop_event.clear()
        self.history.append({"role": "user", "content": user_text})
        self._maybe_summarize_history()

        route = "hybrid"
        if self.pending_task_state:
            route = str(self.pending_task_state.get("route", "hybrid"))
        else:
            status_cb("Thinking")
            router = self.llm.route(self._base_metadata(user_text))
            route = router.route
            self.logger.log("router_result", router.model_dump())

        if route == "chat":
            decision = self.llm.chat(self._base_metadata(user_text))
            reply = decision.response or "Слушаю тебя."
            self._postprocess_decision(decision)
            self._add_assistant_message(reply)
            status_cb("Idle")
            return reply

        reply = self._run_agent_loop(user_text=user_text, route=route, status_cb=status_cb)
        self._add_assistant_message(reply)
        status_cb("Idle")
        return reply

    def _run_agent_loop(self, user_text: str, route: str, status_cb: Callable[[str], None]) -> str:
        task_state = self.pending_task_state or {
            "goal": user_text,
            "route": route,
            "steps_done": 0,
            "last_results": [],
            "stagnation": 0,
        }
        self.pending_task_state = None
        self.awaiting_user = False

        final_reply = "Готово."
        for step in range(CONFIG.agent.max_steps):
            if self.stop_event.is_set():
                return "Остановлено"
            status_cb("Thinking" if step == 0 else "Acting")
            metadata = self._agent_metadata(user_text, task_state)
            decision = self.llm.agent_step(metadata)
            self.logger.log("agent_step", {"step": step + 1, "decision": decision.model_dump(by_alias=True)})
            self._postprocess_decision(decision)

            if decision.ask_user:
                self.pending_task_state = task_state
                self.awaiting_user = True
                return decision.ask_user

            if decision.actions:
                results = self.executor.run_actions(decision.actions)
                task_state["last_results"] = [r.model_dump() for r in results]
                task_state["steps_done"] = int(task_state.get("steps_done", 0)) + 1
                self.logger.log("action_results", task_state["last_results"])
                if all(not r.ok for r in results):
                    msg = results[0].error_message or "действие завершилось с ошибкой"
                    final_reply = f"Не удалось выполнить шаг: {msg}"
                    if decision.continue_ == 1:
                        return final_reply
            elif decision.continue_ == 1:
                task_state["stagnation"] = int(task_state.get("stagnation", 0)) + 1
                if task_state["stagnation"] >= 2:
                    return "Нужны уточнения: я не могу продолжить без дополнительных данных."

            if decision.response:
                final_reply = decision.response
            if decision.continue_ == 0:
                self.awaiting_user = False
                return final_reply

        return "Достигнут лимит шагов. Уточни, что сделать дальше."

    def _agent_metadata(self, user_text: str, task_state: dict[str, Any]) -> dict[str, Any]:
        data = self._base_metadata(user_text)
        data.update({"task_state": task_state})
        return data

    def _base_metadata(self, user_text: str) -> dict[str, Any]:
        tail_size = max(5, self.max_history_messages)
        return {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "memory": self.memory.get_memory_payload(),
            "conversation_summary": self.history_summary,
            "conversation_tail": self.history[-tail_size:],
            "user_input": user_text,
            "awaiting_user": self.awaiting_user,
        }

    def _postprocess_decision(self, decision: LLMDecision) -> None:
        if isinstance(decision.memory_update, dict) and decision.memory_update:
            self.memory.apply_updates(decision.memory_update)
            self.logger.log("memory_update", decision.memory_update)

    def _add_assistant_message(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self.history = self.history[-max(self.max_history_messages * 2, 40) :]

    def _maybe_summarize_history(self) -> None:
        if len(self.history) <= CONFIG.agent.summary_trigger:
            return
        old = self.history[:-self.max_history_messages]
        if not old:
            return
        self.history_summary = " | ".join(f"{x['role']}: {x['content'][:120]}" for x in old[-20:])
        self.history = self.history[-self.max_history_messages :]
