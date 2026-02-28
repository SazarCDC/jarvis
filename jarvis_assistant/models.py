from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Intent = Literal["chat", "question", "action", "noise", "web", "hybrid"]
RouteType = Literal["chat", "question", "web-research", "pc-action", "hybrid"]
ActionType = Literal[
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
]


class ActionSpec(BaseModel):
    type: ActionType
    command: str | None = None
    path: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class LLMDecision(BaseModel):
    intent: Intent
    thought: str
    confidence: float = Field(ge=0.0, le=1.0)
    ask_user: str | None = None
    response: str | None = None
    memory_update: dict[str, Any] | None = None
    actions: list[ActionSpec] = Field(default_factory=list)
    continue_: int = Field(default=0, alias="continue", ge=0, le=1)


class RouterDecision(BaseModel):
    route: RouteType
    thought: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ActionResult(BaseModel):
    action_type: str
    ok: bool
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    paths: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    screenshot_path: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    error_code: str | None = None
