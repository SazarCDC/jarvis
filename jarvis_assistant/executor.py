from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import webbrowser
from pathlib import Path
from threading import Event
from typing import Any
from urllib.parse import urlparse

import pyautogui
import pyperclip
import requests
from PIL import ImageGrab

from jarvis_assistant.config import CONFIG
from jarvis_assistant.models import ActionResult, ActionSpec

try:
    import pygetwindow as gw
except Exception:
    gw = None

try:
    from ddgs import DDGS
except Exception:
    try:
        from duckduckgo_search import DDGS
    except Exception:
        DDGS = None

try:
    import trafilatura
except Exception:
    trafilatura = None


class ActionExecutor:
    def __init__(self, stop_event: Event, command_timeout_sec: int = 120) -> None:
        self.stop_event = stop_event
        self.command_timeout_sec = command_timeout_sec

    def run_actions(self, actions: list[ActionSpec]) -> list[ActionResult]:
        return [self._run_single(action) for action in actions]

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
            "audio": self._audio,
            "web_search": self._web_search,
            "web_fetch": self._web_fetch,
            "web_extract": self._web_extract,
            "monitor": self._monitor,
        }
        handler = handlers.get(action.type)
        if handler is None:
            return ActionResult(action_type=action.type, ok=False, error_message="Неизвестный action type", error_code="UNKNOWN_ACTION")
        try:
            if self.stop_event.is_set():
                return ActionResult(action_type=action.type, ok=False, error_message="Остановлено пользователем", error_code="STOPPED")
            return handler(action)
        except pyautogui.FailSafeException:
            return ActionResult(action_type=action.type, ok=False, error_message="PyAutoGUI FailSafe сработал (курсор в углу).", error_code="PYAUTOGUI_FAILSAFE")
        except Exception as exc:
            return ActionResult(action_type=action.type, ok=False, error_message=str(exc), error_code="ACTION_ERROR")

    def _cmd(self, action: ActionSpec) -> ActionResult:
        cp = subprocess.run(action.command or "", shell=True, capture_output=True, text=True, timeout=self.command_timeout_sec)
        return ActionResult(action_type="cmd", ok=cp.returncode == 0, stdout=cp.stdout, stderr=cp.stderr, exit_code=cp.returncode)

    def _powershell(self, action: ActionSpec) -> ActionResult:
        cp = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", action.command or ""], capture_output=True, text=True, timeout=self.command_timeout_sec)
        return ActionResult(action_type="powershell", ok=cp.returncode == 0, stdout=cp.stdout, stderr=cp.stderr, exit_code=cp.returncode)

    @staticmethod
    def _launch(action: ActionSpec) -> ActionResult:
        target = (action.path or action.command or "").strip()
        if not target:
            return ActionResult(action_type="launch", ok=False, error_message="Не указан путь", error_code="EMPTY_TARGET")
        if any(x in target.lower() for x in ["&&", "|", "taskkill", "powershell "]):
            return ActionResult(action_type="launch", ok=False, error_message="Похоже на shell-команду, используйте cmd/powershell.", error_code="LIKELY_COMMAND")
        try:
            os.startfile(target)
            return ActionResult(action_type="launch", ok=True, paths=[target])
        except FileNotFoundError:
            return ActionResult(action_type="launch", ok=False, error_message=f"Не найдено: {target}", error_code="FILE_NOT_FOUND")

    @staticmethod
    def _search(action: ActionSpec) -> ActionResult:
        root = Path(action.args.get("root", "."))
        pattern = action.args.get("pattern", "*")
        depth = int(action.args.get("max_depth", 3))
        limit = int(action.args.get("limit", 100))
        matches: list[str] = []
        start_depth = len(root.resolve().parts)
        for found in root.rglob(pattern):
            if len(found.resolve().parts) - start_depth <= depth:
                matches.append(str(found))
            if len(matches) >= limit:
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
        keys = action.args.get("keys", [])
        if isinstance(keys, str):
            keys = [keys]
        if mode == "type":
            pyautogui.write(str(action.args.get("text", "")), interval=float(action.args.get("interval", 0.02)))
        elif mode == "hotkey":
            pyautogui.hotkey(*keys)
        elif mode == "press":
            pyautogui.press(keys[0] if keys else str(action.args.get("key", "enter")))
        elif mode == "keydown":
            pyautogui.keyDown(keys[0] if keys else str(action.args.get("key", "")))
        elif mode == "keyup":
            pyautogui.keyUp(keys[0] if keys else str(action.args.get("key", "")))
        else:
            return ActionResult(action_type="keyboard", ok=False, error_message=f"Неподдерживаемый mode={mode}", error_code="BAD_KEYBOARD_MODE")
        return ActionResult(action_type="keyboard", ok=True)

    @staticmethod
    def _mouse(action: ActionSpec) -> ActionResult:
        event = str(action.args.get("event", "click")).lower()
        x = action.args.get("x")
        y = action.args.get("y")
        target_monitor = str(action.args.get("monitor", "")).lower()
        if target_monitor in {"second", "2", "второй"} and (x is None or y is None):
            return ActionResult(action_type="mouse", ok=False, error_message="PyAutoGUI ограничен в multi-monitor. Укажи x/y или используй главный монитор.", error_code="MULTI_MONITOR_LIMIT")
        if x is not None and y is not None:
            pyautogui.moveTo(int(x), int(y), duration=float(action.args.get("duration", 0.1)))
        button = str(action.args.get("button", "left"))
        clicks = int(action.args.get("clicks", 1))
        if event == "click":
            pyautogui.click(button=button, clicks=clicks)
        elif event == "doubleclick":
            pyautogui.doubleClick(button=button)
        elif event == "rightclick":
            pyautogui.rightClick()
        elif event == "scroll":
            pyautogui.scroll(int(action.args.get("amount", -500)))
        else:
            return ActionResult(action_type="mouse", ok=False, error_message=f"Неподдерживаемое событие: {event}", error_code="BAD_MOUSE_EVENT")
        return ActionResult(action_type="mouse", ok=True)

    @staticmethod
    def _window(action: ActionSpec) -> ActionResult:
        title = str(action.args.get("title", "") or "").strip()
        command = str(action.args.get("command", "") or "").strip().lower()
        if command not in {"activate", "minimize", "maximize", "close"}:
            return ActionResult(action_type="window", ok=False, error_message="command должен быть activate/minimize/maximize/close", error_code="BAD_WINDOW_COMMAND")
        if gw is None:
            return ActionResult(action_type="window", ok=False, error_message="pygetwindow недоступен", error_code="MISSING_DEPENDENCY")
        windows = gw.getWindowsWithTitle(title) if title else []
        if not windows:
            return ActionResult(action_type="window", ok=False, error_message=f"Окно '{title}' не найдено", error_code="WINDOW_NOT_FOUND")
        w = windows[0]
        if command == "activate":
            w.activate()
        elif command == "minimize":
            w.minimize()
        elif command == "maximize":
            w.maximize()
        else:
            w.close()
        return ActionResult(action_type="window", ok=True)

    @staticmethod
    def _screenshot(action: ActionSpec) -> ActionResult:
        path = Path(action.path or action.args.get("path", f"screenshots/screen_{int(time.time())}.png"))
        path.parent.mkdir(parents=True, exist_ok=True)
        all_screens = bool(action.args.get("all_screens", False))
        img = ImageGrab.grab(all_screens=all_screens)
        img.save(path)
        return ActionResult(action_type="screenshot", ok=True, screenshot_path=str(path), files=[str(path)])

    @staticmethod
    def _clipboard(action: ActionSpec) -> ActionResult:
        mode = action.args.get("mode", "read")
        if mode == "write":
            pyperclip.copy(action.args.get("text", ""))
            return ActionResult(action_type="clipboard", ok=True)
        return ActionResult(action_type="clipboard", ok=True, data={"text": pyperclip.paste()})

    def _wait(self, action: ActionSpec) -> ActionResult:
        seconds = float(action.args.get("seconds", 1))
        until = time.time() + seconds
        while time.time() < until:
            if self.stop_event.is_set():
                return ActionResult(action_type="wait", ok=False, error_message="Остановлено пользователем", error_code="STOPPED")
            time.sleep(0.1)
        return ActionResult(action_type="wait", ok=True)

    @staticmethod
    def _browser(action: ActionSpec) -> ActionResult:
        url = (action.command or action.args.get("url") or "").strip()
        if not url:
            return ActionResult(action_type="browser", ok=False, error_message="Пустой URL", error_code="EMPTY_URL")
        webbrowser.open(url)
        return ActionResult(action_type="browser", ok=True, data={"url": url})

    def _audio(self, action: ActionSpec) -> ActionResult:
        cmd = str(action.args.get("command", "")).lower()
        if not cmd:
            return ActionResult(action_type="audio", ok=False, error_message="Не указан command", error_code="BAD_AUDIO_COMMAND")

        pre_check = self._powershell(ActionSpec(type="powershell", command="Get-Module -ListAvailable -Name AudioDeviceCmdlets"))
        if pre_check.exit_code != 0 or not (pre_check.stdout or "").strip():
            return ActionResult(
                action_type="audio",
                ok=False,
                error_message="PowerShell модуль AudioDeviceCmdlets не установлен.",
                error_code="AUDIO_MODULE_MISSING",
                data={"hint": "Install-Module AudioDeviceCmdlets -Scope CurrentUser"},
            )

        ps = "Import-Module AudioDeviceCmdlets; "
        if cmd == "set_volume":
            ps += f"Set-AudioDevice -PlaybackVolume {int(action.args.get('percent', 50))}"
        elif cmd == "change_volume":
            ps += f"$v=(Get-AudioDevice -PlaybackVolume).Volume; Set-AudioDevice -PlaybackVolume ($v+{int(action.args.get('delta_percent', 5))})"
        elif cmd == "mute":
            ps += f"Set-AudioDevice -Mute ${str(bool(action.args.get('value', True))).lower()}"
        elif cmd == "list_devices":
            kind = "Playback" if str(action.args.get("kind", "playback")).lower() == "playback" else "Recording"
            ps += f"Get-AudioDevice -List | Where-Object {{$_.Type -match '{kind}'}} | ConvertTo-Json -Depth 4"
        elif cmd == "set_default_device":
            needle = str(action.args.get("name_contains", "")).lower()
            ps += f"$d=Get-AudioDevice -List | Where-Object {{$_.Name.ToLower().Contains('{needle}')}} | Select-Object -First 1; if($d){{Set-AudioDevice -Id $d.ID}} else {{throw 'device not found'}}"
        else:
            return ActionResult(action_type="audio", ok=False, error_message=f"Неизвестная audio команда: {cmd}", error_code="BAD_AUDIO_COMMAND")

        res = self._powershell(ActionSpec(type="powershell", command=ps))
        data = {}
        if cmd == "list_devices" and res.stdout:
            try:
                data["devices"] = json.loads(res.stdout)
            except Exception:
                data["raw"] = res.stdout
        return ActionResult(action_type="audio", ok=res.ok, stdout=res.stdout, stderr=res.stderr, exit_code=res.exit_code, data=data, error_message=res.error_message, error_code=res.error_code)

    def _web_search(self, action: ActionSpec) -> ActionResult:
        if not CONFIG.web.enabled:
            return ActionResult(action_type="web_search", ok=False, error_message="Web disabled in settings", error_code="WEB_DISABLED")
        if DDGS is None:
            return ActionResult(action_type="web_search", ok=False, error_message="Библиотека веб-поиска не установлена (установите пакет ddgs)", error_code="WEB_SEARCH_LIB_MISSING")
        query = (action.command or action.args.get("query") or "").strip()
        if not query:
            return ActionResult(action_type="web_search", ok=False, error_message="Пустой query", error_code="EMPTY_QUERY")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region=CONFIG.web.region, max_results=CONFIG.web.max_results))
        compact = [{"title": i.get("title"), "href": i.get("href"), "body": i.get("body")} for i in results]
        return ActionResult(action_type="web_search", ok=True, data={"results": compact})

    def _web_fetch(self, action: ActionSpec) -> ActionResult:
        url = (action.command or action.args.get("url") or "").strip()
        if not url:
            return ActionResult(action_type="web_fetch", ok=False, error_message="Пустой URL", error_code="EMPTY_URL")
        host = (urlparse(url).hostname or "").lower()
        trust_env = host not in {"localhost", "127.0.0.1"}
        s = requests.Session()
        s.trust_env = trust_env
        r = s.get(url, timeout=CONFIG.web.timeout_sec)
        text = (r.text or "")[: CONFIG.web.fetch_max_chars]
        return ActionResult(action_type="web_fetch", ok=r.ok, data={"url": url, "status": r.status_code, "content_type": r.headers.get("content-type", ""), "html": text})

    def _web_extract(self, action: ActionSpec) -> ActionResult:
        html = str(action.args.get("html") or "")
        url = str(action.args.get("url") or action.command or "")
        if not html and url:
            fetch = self._web_fetch(ActionSpec(type="web_fetch", command=url, args={}))
            if not fetch.ok:
                return ActionResult(action_type="web_extract", ok=False, error_message=fetch.error_message, error_code=fetch.error_code)
            html = str(fetch.data.get("html", ""))
        if not html:
            return ActionResult(action_type="web_extract", ok=False, error_message="Нет HTML для извлечения", error_code="EMPTY_HTML")

        if trafilatura is not None:
            extracted = trafilatura.extract(html, output_format="txt", include_comments=False, include_tables=False) or ""
            title = trafilatura.extract_metadata(html).title if trafilatura.extract_metadata(html) else ""
        else:
            extracted = " ".join(html.split())
            title = ""
        return ActionResult(action_type="web_extract", ok=True, data={"title": title, "clean_text": extracted[: CONFIG.web.extract_max_chars]})

    @staticmethod
    def _monitor(action: ActionSpec) -> ActionResult:
        try:
            size = pyautogui.size()
            return ActionResult(action_type="monitor", ok=True, data={"primary": {"width": size.width, "height": size.height}, "note": "PyAutoGUI ограничен для multi-monitor"})
        except Exception as exc:
            return ActionResult(action_type="monitor", ok=False, error_message=str(exc), error_code="MONITOR_QUERY_FAILED")
