from __future__ import annotations

import os
import subprocess
import time
import webbrowser
from pathlib import Path
from threading import Event

import pyautogui
import pyperclip
from PIL import ImageGrab

from jarvis_assistant.models import ActionResult, ActionSpec

try:
    import pygetwindow as gw
except Exception:
    gw = None


class ActionExecutor:
    def __init__(self, stop_event: Event, command_timeout_sec: int = 120) -> None:
        self.stop_event = stop_event
        self.command_timeout_sec = command_timeout_sec

    def run_actions(self, actions: list[ActionSpec]) -> list[ActionResult]:
        results: list[ActionResult] = []
        for action in actions:
            if self.stop_event.is_set():
                results.append(ActionResult(action_type=action.type, ok=False, error_message="Остановлено пользователем"))
                break
            results.append(self._run_single(action))
        return results

    def _run_single(self, action: ActionSpec) -> ActionResult:
        handlers = {
            "cmd": self._cmd,
            "powershell": self._powershell,
            "launch": self._launch,
            "search": self._search,
            "write_file": self._write_file,
            "read_file": self._read_file,
            "keyboard": self._keyboard,
            "mouse": self._mouse,
            "window": self._window,
            "screenshot": self._screenshot,
            "clipboard": self._clipboard,
            "wait": self._wait,
            "browser": self._browser,
        }
        handler = handlers.get(action.type)
        if handler is None:
            return ActionResult(action_type=action.type, ok=False, error_message="Неизвестный action type")
        try:
            return handler(action)
        except Exception as exc:
            return ActionResult(action_type=action.type, ok=False, error_message=str(exc))

    def _cmd(self, action: ActionSpec) -> ActionResult:
        command = action.command or ""
        if action.args.get("as_bat"):
            bat_path = Path(action.args.get("bat_path", "temp_action.bat"))
            bat_path.write_text(command, encoding="utf-8")
            command = str(bat_path)
        cp = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=self.command_timeout_sec)
        return ActionResult(action_type="cmd", ok=cp.returncode == 0, stdout=cp.stdout, stderr=cp.stderr, exit_code=cp.returncode)

    def _powershell(self, action: ActionSpec) -> ActionResult:
        cp = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", action.command or ""],
            capture_output=True,
            text=True,
            timeout=self.command_timeout_sec,
        )
        return ActionResult(action_type="powershell", ok=cp.returncode == 0, stdout=cp.stdout, stderr=cp.stderr, exit_code=cp.returncode)

    @staticmethod
    def _launch(action: ActionSpec) -> ActionResult:
        target = action.path or action.command or ""
        os.startfile(target)
        return ActionResult(action_type="launch", ok=True, paths=[target])

    @staticmethod
    def _search(action: ActionSpec) -> ActionResult:
        root = Path(action.args.get("root", "."))
        pattern = action.args.get("pattern", "*")
        depth = int(action.args.get("max_depth", 3))
        matches: list[str] = []
        start_depth = len(root.resolve().parts)
        for found in root.rglob(pattern):
            if len(found.resolve().parts) - start_depth <= depth:
                matches.append(str(found))
            if len(matches) >= int(action.args.get("limit", 100)):
                break
        return ActionResult(action_type="search", ok=True, paths=matches)

    @staticmethod
    def _write_file(action: ActionSpec) -> ActionResult:
        path = Path(action.path or action.args.get("path"))
        content = action.command or action.args.get("content", "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ActionResult(action_type="write_file", ok=True, files=[str(path)])

    @staticmethod
    def _read_file(action: ActionSpec) -> ActionResult:
        path = Path(action.path or action.args.get("path"))
        text = path.read_text(encoding="utf-8")
        return ActionResult(action_type="read_file", ok=True, files=[str(path)], data={"content": text[:8000]})

    @staticmethod
    def _keyboard(action: ActionSpec) -> ActionResult:
        mode = action.args.get("mode", "hotkey")
        if mode == "type":
            pyautogui.write(action.args.get("text", ""), interval=float(action.args.get("interval", 0.02)))
        else:
            keys = action.args.get("keys", [])
            if isinstance(keys, str):
                keys = [keys]
            pyautogui.hotkey(*keys)
        return ActionResult(action_type="keyboard", ok=True)

    @staticmethod
    def _mouse(action: ActionSpec) -> ActionResult:
        event = action.args.get("event", "click")
        x = action.args.get("x")
        y = action.args.get("y")
        if x is not None and y is not None:
            pyautogui.moveTo(x, y, duration=float(action.args.get("duration", 0.1)))
        if event == "click":
            pyautogui.click()
        elif event == "doubleclick":
            pyautogui.doubleClick()
        elif event == "rightclick":
            pyautogui.rightClick()
        elif event == "scroll":
            pyautogui.scroll(int(action.args.get("amount", -500)))
        return ActionResult(action_type="mouse", ok=True)

    @staticmethod
    def _window(action: ActionSpec) -> ActionResult:
        if gw is None:
            return ActionResult(action_type="window", ok=False, error_message="pygetwindow недоступен")
        title = action.args.get("title", "")
        command = action.args.get("command", "activate")
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            return ActionResult(action_type="window", ok=False, error_message=f"Окно с заголовком '{title}' не найдено")
        window = windows[0]
        if command == "activate":
            window.activate()
        elif command == "minimize":
            window.minimize()
        elif command == "maximize":
            window.maximize()
        elif command == "close":
            window.close()
        return ActionResult(action_type="window", ok=True)

    @staticmethod
    def _screenshot(action: ActionSpec) -> ActionResult:
        path = Path(action.path or action.args.get("path", f"screenshots/screen_{int(time.time())}.png"))
        path.parent.mkdir(parents=True, exist_ok=True)
        img = ImageGrab.grab()
        img.save(path)
        return ActionResult(action_type="screenshot", ok=True, screenshot_path=str(path), files=[str(path)])

    @staticmethod
    def _clipboard(action: ActionSpec) -> ActionResult:
        mode = action.args.get("mode", "read")
        if mode == "write":
            pyperclip.copy(action.args.get("text", ""))
            return ActionResult(action_type="clipboard", ok=True)
        text = pyperclip.paste()
        return ActionResult(action_type="clipboard", ok=True, data={"text": text})

    def _wait(self, action: ActionSpec) -> ActionResult:
        seconds = float(action.args.get("seconds", 1))
        slice_s = 0.1
        waited = 0.0
        while waited < seconds:
            if self.stop_event.is_set():
                return ActionResult(action_type="wait", ok=False, error_message="Остановлено пользователем")
            time.sleep(slice_s)
            waited += slice_s
        return ActionResult(action_type="wait", ok=True)

    @staticmethod
    def _browser(action: ActionSpec) -> ActionResult:
        url = action.command or action.args.get("url") or ""
        webbrowser.open(url)
        return ActionResult(action_type="browser", ok=True, data={"url": url})
