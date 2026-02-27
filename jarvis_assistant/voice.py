from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Callable

from jarvis_assistant.assistant_core import JarvisAssistant
from jarvis_assistant.logger import JsonlLogger


class VoiceController:
    WAKE_ALIASES = ("джарвис", "джарвиз", "жарвис", "jarvis")

    def __init__(
        self,
        assistant: JarvisAssistant,
        logger: JsonlLogger,
        on_user_text: Callable[[str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        is_busy: Callable[[], bool],
        on_stopped: Callable[[], None] | None = None,
    ) -> None:
        self.assistant = assistant
        self.logger = logger
        self.on_user_text = on_user_text
        self.on_status = on_status
        self.on_error = on_error
        self.is_busy = is_busy
        self.on_stopped = on_stopped

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._tts_lock = threading.Lock()
        self._tts_engine = None

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="voice-loop")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._stop_tts()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def speak(self, text: str) -> None:
        cleaned = sanitize_for_tts(text)
        if not cleaned:
            return
        try:
            self._speak_internal(cleaned)
        except Exception as exc:
            self.logger.log("voice_error", {"stage": "tts", "error": str(exc)})
            self.on_error(f"Ошибка синтеза речи: {exc}")
        finally:
            self.on_status("Listening" if self.is_running() else "Idle")

    def _run_loop(self) -> None:
        try:
            import speech_recognition as sr
        except Exception as exc:
            self._fatal_voice_error(f"Voice mode недоступен: не установлен speech_recognition ({exc}).")
            return

        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1.0
        recognizer.non_speaking_duration = 0.5
        energy_override = os.getenv("JARVIS_STT_ENERGY_THRESHOLD")
        if energy_override:
            try:
                recognizer.energy_threshold = max(1, int(float(energy_override)))
                recognizer.dynamic_energy_threshold = False
            except ValueError:
                self.logger.log(
                    "voice_error",
                    {
                        "stage": "stt_config",
                        "error": f"Invalid JARVIS_STT_ENERGY_THRESHOLD: {energy_override}",
                    },
                )

        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                self.on_status("Listening")
                while not self._stop_event.is_set():
                    text = self._listen_for_wake_word(recognizer, source)
                    if not text:
                        continue
                    if not self._contains_wake_word(text):
                        continue

                    self.logger.log("wake_word_detected", {"text": text})
                    self.on_status("Heard wake word")

                    if self.is_busy():
                        self._speak_internal("Подожди секунду")
                        self.on_status("Listening")
                        continue

                    self._speak_internal("Слушаю")
                    command = self._listen_for_command(recognizer, source)
                    if not command:
                        self._speak_internal("Да?")
                        command = self._listen_for_command(recognizer, source)
                        if not command:
                            self.logger.log("command_not_heard", {"attempts": 2})

                    if command:
                        self.logger.log("stt_text", {"text": command})
                        self.on_user_text(command)
                    self.on_status("Listening")
        except OSError as exc:
            self._fatal_voice_error(f"Не удалось получить доступ к микрофону: {exc}")
            return
        except Exception as exc:
            self._fatal_voice_error(f"Ошибка voice loop: {exc}")
            return
        finally:
            self.on_status("Idle")
            if self.on_stopped:
                self.on_stopped()

    def _listen_for_wake_word(self, recognizer, source) -> str | None:
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
        except Exception as exc:
            self._log_stt_error(stage="wake_word", exc=exc)
            return None
        return self._recognize_ru(recognizer, audio, stage="wake_word")

    def _listen_for_command(self, recognizer, source) -> str | None:
        if self._stop_event.is_set():
            return None
        try:
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=12)
        except Exception as exc:
            self._log_stt_error(stage="command", exc=exc)
            return None
        return self._recognize_ru(recognizer, audio, stage="command")

    def _recognize_ru(self, recognizer, audio, stage: str) -> str | None:
        try:
            text = recognizer.recognize_google(audio, language="ru-RU")
            return text.strip()
        except Exception as exc:
            err_name = self._log_stt_error(stage=stage, exc=exc)
            if err_name == "RequestError":
                self.on_error("Ошибка STT: проверь интернет-соединение для распознавания речи.")
                time.sleep(1)
            return None

    @classmethod
    def _contains_wake_word(cls, text: str) -> bool:
        lowered = text.lower()
        return any(alias in lowered for alias in cls.WAKE_ALIASES)

    def _speak_internal(self, text: str) -> None:
        if self._stop_event.is_set():
            return
        self.on_status("Speaking")
        self.logger.log("tts_started", {"text": text})
        with self._tts_lock:
            engine = self._init_tts_engine()
            if engine is None:
                self.logger.log("tts_error", {"reason": "pyttsx3 init failed"})
                return

            engine.stop()
            engine.say(text)
            run_error: Exception | None = None

            def _run_and_wait() -> None:
                nonlocal run_error
                try:
                    engine.runAndWait()
                except Exception as exc:
                    run_error = exc

            worker = threading.Thread(target=_run_and_wait, daemon=True, name="tts-runner")
            worker.start()
            worker.join(timeout=8)

            if worker.is_alive():
                self.logger.log("tts_error", {"reason": "runAndWait timeout", "timeout_s": 8})
                try:
                    engine.stop()
                except Exception:
                    pass
                self._tts_engine = None
                return

            if run_error is not None:
                self.logger.log("tts_error", {"reason": "runAndWait failed", "error": str(run_error)})
                try:
                    engine.stop()
                except Exception:
                    pass
                self._tts_engine = None
                return

        self.logger.log("tts_finished", {"text": text})

    def _init_tts_engine(self):
        if self._tts_engine is not None:
            return self._tts_engine
        try:
            import pyttsx3

            self._tts_engine = pyttsx3.init()
            return self._tts_engine
        except Exception as exc:
            self.logger.log("tts_error", {"reason": "pyttsx3 init exception", "error": str(exc)})
            self._tts_engine = None
            return None

    def _stop_tts(self) -> None:
        with self._tts_lock:
            if self._tts_engine is not None:
                self._tts_engine.stop()

    def _fatal_voice_error(self, message: str) -> None:
        self.logger.log("voice_error", {"stage": "fatal", "error": message})
        self.on_error(message)
        self._stop_event.set()

    def _log_stt_error(self, stage: str, exc: Exception) -> str:
        err_name = exc.__class__.__name__
        self.logger.log(
            "stt_error",
            {
                "stage": stage,
                "error": err_name,
                "details": str(exc),
            },
        )
        return err_name


def sanitize_for_tts(text: str, max_len: int = 500) -> str:
    cleaned = text.replace("⚠️", "Внимание").replace("⚠", "Внимание")
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"[A-Za-z]:\\[^\s]+", "", cleaned)
    cleaned = re.sub(r"/[\w\-./]{12,}", "", cleaned)
    cleaned = re.sub(r"\{.*?\}", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\[.*?\]", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned.startswith("{") or cleaned.startswith("["):
        try:
            json.loads(cleaned)
            return ""
        except Exception:
            pass

    return cleaned[:max_len].strip()
