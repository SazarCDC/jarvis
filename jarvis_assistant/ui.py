from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import scrolledtext

from jarvis_assistant.assistant_core import JarvisAssistant
from jarvis_assistant.logger import JsonlLogger
from jarvis_assistant.voice import VoiceController


class JarvisApp(tk.Tk):
    def __init__(self, assistant: JarvisAssistant, logger: JsonlLogger) -> None:
        super().__init__()
        self.assistant = assistant
        self.logger = logger
        self.title("Jarvis Desktop Assistant")
        self.geometry("900x650")
        self.minsize(750, 500)

        self.status_var = tk.StringVar(value="Idle")
        self.input_var = tk.StringVar()
        self.voice_enabled = tk.BooleanVar(value=False)
        self.event_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.voice_controller = VoiceController(
            assistant=self.assistant,
            logger=self.logger,
            on_user_text=self._on_voice_user_text,
            on_status=self._on_voice_status,
            on_error=self._on_voice_error,
            is_busy=self._is_busy,
            on_stopped=self._on_voice_stopped,
        )

        self._build_widgets()
        self.after(100, self._poll_events)

    def _build_widgets(self) -> None:
        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 0))

        tk.Label(top, text="Status:", font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Label(top, textvariable=self.status_var, fg="#0b69d0").pack(side="left", padx=(6, 0))

        self.voice_btn = tk.Button(top, text="Voice: OFF", width=12, command=self.on_toggle_voice)
        self.voice_btn.pack(side="right")

        self.chat = scrolledtext.ScrolledText(self, wrap=tk.WORD, state="disabled", font=("Segoe UI", 10))
        self.chat.pack(fill="both", expand=True, padx=10, pady=10)

        bottom = tk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        entry = tk.Entry(bottom, textvariable=self.input_var, font=("Segoe UI", 11))
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        entry.bind("<Return>", lambda _: self.on_send())

        send_btn = tk.Button(bottom, text="Execute / Send", width=16, command=self.on_send)
        send_btn.pack(side="left", padx=(0, 6))
        stop_btn = tk.Button(bottom, text="STOP", width=10, fg="white", bg="#d9534f", command=self.on_stop)
        stop_btn.pack(side="left")

    def on_send(self) -> None:
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self._submit_user_text(text)

    def on_toggle_voice(self) -> None:
        if self.voice_enabled.get():
            self.voice_enabled.set(False)
            self.voice_btn.configure(text="Voice: OFF")
            self.voice_controller.stop()
            if self.status_var.get() in {"Listening", "Heard wake word", "Speaking"}:
                self.status_var.set("Idle")
            self._append("System", "Voice mode выключен.")
            return

        self.voice_enabled.set(True)
        self.voice_btn.configure(text="Voice: ON")
        self.voice_controller.start()
        self._append("System", "Voice mode включен. Ожидание wake word: 'джарвис'.")

    def on_stop(self) -> None:
        self.assistant.stop()
        self.voice_controller.stop()
        self.status_var.set("Idle")
        self._append("System", "Запрошена немедленная остановка.")

    def _run_assistant(self, text: str) -> None:
        def status_cb(new_status: str) -> None:
            self.event_queue.put(("status", new_status))

        try:
            reply = self.assistant.process_user_message(text, status_cb=status_cb)
            self.event_queue.put(("assistant", reply))
            if self.voice_enabled.get() and reply:
                self.voice_controller.speak(reply)
        except Exception as exc:
            self.event_queue.put(("assistant", f"Ошибка: {exc}"))
            self.event_queue.put(("status", "Idle"))

    def _submit_user_text(self, text: str) -> None:
        if self._is_busy():
            self._append("System", "Подожди завершения текущего цикла или нажми STOP.")
            return
        self._append("User", text)
        self.worker = threading.Thread(target=self._run_assistant, args=(text,), daemon=True)
        self.worker.start()

    def _on_voice_user_text(self, text: str) -> None:
        self.event_queue.put(("voice_user", text))

    def _on_voice_status(self, status: str) -> None:
        self.event_queue.put(("status", status))

    def _on_voice_error(self, err_text: str) -> None:
        self.event_queue.put(("system", err_text))

    def _on_voice_stopped(self) -> None:
        self.event_queue.put(("voice_stopped", ""))

    def _is_busy(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def _poll_events(self) -> None:
        while True:
            try:
                etype, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if etype == "status":
                self.status_var.set(payload)
            elif etype == "assistant":
                self._append("Jarvis", payload)
            elif etype == "voice_user":
                self._submit_user_text(payload)
            elif etype == "system":
                self._append("System", payload)
            elif etype == "voice_stopped" and self.voice_enabled.get():
                self.voice_enabled.set(False)
                self.voice_btn.configure(text="Voice: OFF")
                if self.status_var.get() in {"Listening", "Heard wake word", "Speaking"}:
                    self.status_var.set("Idle")
        self.after(100, self._poll_events)

    def _append(self, role: str, text: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert(tk.END, f"{role}: {text}\n\n")
        self.chat.see(tk.END)
        self.chat.configure(state="disabled")
