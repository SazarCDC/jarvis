from __future__ import annotations

import os
import queue
import re
import threading
import time
from typing import Callable

import numpy as np

from jarvis_assistant.assistant_core import JarvisAssistant
from jarvis_assistant.logger import JsonlLogger


class VoiceController:
    SAMPLE_RATE = 16000
    FRAME_MS = 32

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

        self._tts_queue: queue.Queue[str | tuple[str, float] | None] = queue.Queue()
        self._tts_thread: threading.Thread | None = None
        self._tts_stop = threading.Event()
        self._tts_engine = None
        self._volume_0_1 = self._read_volume_from_env()
        self._tts_ready = threading.Event()
        self._tts_watchdog_sec = self._float_env("JARVIS_TTS_WATCHDOG_SEC", 20.0)
        self._status_lock = threading.Lock()
        self._current_status = "Idle"

        self._porcupine_sensitivity = min(1.0, max(0.0, self._float_env("JARVIS_PORCUPINE_SENSITIVITY", 0.7)))
        self._wake_debug = self._int_env("JARVIS_WAKE_DEBUG", 0) == 1
        self._wake_cooldown_sec = max(0.0, self._float_env("JARVIS_WAKE_COOLDOWN", 1.0))
        self._command_pre_roll_ms = max(0, self._int_env("JARVIS_COMMAND_PRE_ROLL_MS", 200))
        self._command_start_timeout = max(0.3, self._float_env("JARVIS_COMMAND_START_TIMEOUT", 2.5))
        self._selected_device_label = "unknown"

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self.start_tts()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="voice-loop")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.stop_tts()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start_tts(self) -> None:
        if self._tts_thread and self._tts_thread.is_alive():
            return
        self._tts_stop.clear()
        self._tts_ready.clear()
        self._tts_thread = threading.Thread(target=self._tts_loop, daemon=True, name="tts-loop")
        self._tts_thread.start()

    def stop_tts(self) -> None:
        self._tts_stop.set()
        self._clear_tts_queue()
        self._enqueue_tts(("__STOP__", 0.0))
        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
            except Exception as exc:
                self.logger.log("tts_error", {"error": str(exc), "stage": "stop"})
        self._enqueue_tts(None)

    def set_volume(self, volume_0_1: float) -> None:
        self._volume_0_1 = min(1.0, max(0.0, float(volume_0_1)))
        self.start_tts()
        self._enqueue_tts(("__SET_VOLUME__", self._volume_0_1))

    def speak(self, text: str) -> None:
        cleaned = sanitize_for_tts(text)
        if not cleaned or self._stop_event.is_set():
            return
        self.start_tts()
        self._enqueue_tts(cleaned)

    def _enqueue_tts(self, item: str | tuple[str, float] | None) -> None:
        try:
            self._tts_queue.put_nowait(item)
        except queue.Full:
            self.logger.log("tts_error", {"error": "tts_queue_full"})

    def _clear_tts_queue(self) -> None:
        while True:
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break

    def _tts_loop(self) -> None:
        engine = None
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("volume", self._volume_0_1)
            self._tts_engine = engine
            self._tts_ready.set()

            while not self._tts_stop.is_set():
                item = self._tts_queue.get()
                if item is None:
                    break

                if isinstance(item, tuple) and item[0] == "__SET_VOLUME__":
                    try:
                        engine.setProperty("volume", float(item[1]))
                    except Exception as exc:
                        self.logger.log("tts_error", {"error": str(exc), "stage": "set_volume"})
                    continue

                if isinstance(item, tuple) and item[0] == "__STOP__":
                    try:
                        engine.stop()
                    except Exception as exc:
                        self.logger.log("tts_error", {"error": str(exc), "stage": "stop"})
                    continue

                text = str(item)
                started_at = time.monotonic()
                try:
                    self._set_status("Speaking")
                    self.logger.log("tts_started", {"text": text[:120]})
                    engine.stop()
                    engine.say(text)
                    engine.runAndWait()
                    if time.monotonic() - started_at > self._tts_watchdog_sec:
                        self.logger.log(
                            "tts_watchdog_timeout",
                            {"duration_sec": round(time.monotonic() - started_at, 2), "text": text[:120]},
                        )
                    self.logger.log("tts_finished", {"text": text[:120]})
                except Exception as exc:
                    self.logger.log("tts_error", {"error": str(exc)})
                finally:
                    self._set_status("Listening" if self.is_running() and not self._stop_event.is_set() else "Idle")
        except Exception as exc:
            self.logger.log("tts_error", {"error": str(exc), "stage": "init"})
            self.on_error(f"Ошибка синтеза речи: {exc}")
        finally:
            self._tts_engine = None
            self._tts_ready.clear()
            try:
                if engine is not None:
                    engine.stop()
            except Exception:
                pass

    def _run_loop(self) -> None:
        recorder = None
        porcupine = None
        try:
            import pvporcupine
            from faster_whisper import WhisperModel
            from pvrecorder import PvRecorder

            access_key = os.getenv("JARVIS_PICOVOICE_ACCESS_KEY", "").strip()
            if not access_key:
                self.on_error("Wake word недоступен: не задан JARVIS_PICOVOICE_ACCESS_KEY")
                return

            model_path = os.getenv("JARVIS_PICOVOICE_MODEL_PATH", "").strip()
            if not model_path or not os.path.isfile(model_path):
                self.on_error("Wake word недоступен: файл .ppn не найден")
                return

            command_max_sec = self._float_env("JARVIS_COMMAND_MAX_SEC", 10.0)
            silence_ms = self._int_env("JARVIS_COMMAND_SILENCE_MS", 900, min_value=300, max_value=3000)
            rms_threshold = self._float_env("JARVIS_VAD_RMS_THRESHOLD", 700.0)
            whisper_model_name = os.getenv("JARVIS_WHISPER_MODEL", "small").strip() or "small"
            beam_size = self._int_env("JARVIS_WHISPER_BEAM_SIZE", 1, min_value=1, max_value=3)
            device_index = self._int_env("JARVIS_AUDIO_DEVICE_INDEX", 0)
            available_devices = PvRecorder.get_available_devices()
            if not available_devices:
                self.on_error("Wake word недоступен: PvRecorder не нашел ни одного устройства ввода")
                return

            if device_index != -1 and (device_index < 0 or device_index >= len(available_devices)):
                self.logger.log(
                    "voice_error",
                    {
                        "stage": "config",
                        "error": f"JARVIS_AUDIO_DEVICE_INDEX={device_index} out of range, fallback to 0",
                    },
                )
                device_index = 0
            device_name = "auto" if device_index == -1 else available_devices[device_index]
            self._selected_device_label = f"{device_index}: {device_name}"

            porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[model_path],
                sensitivities=[self._porcupine_sensitivity],
            )
            recorder = PvRecorder(
                device_index=device_index,
                frame_length=porcupine.frame_length,
            )
            whisper_model = WhisperModel(whisper_model_name, device="cpu", compute_type="int8")

            self.logger.log("wake_backend_selected", {"backend": "porcupine", "model": ".ppn"})
            self.logger.log("voice_started", {})
            self.logger.log(
                "voice_wake_config",
                {
                    "sensitivity": self._porcupine_sensitivity,
                    "device_index": device_index,
                    "device_name": device_name,
                },
            )


            self._set_status("Listening")
            recorder.start()
            last_wake_at = 0.0
            last_debug_at = 0.0

            while not self._stop_event.is_set():
                pcm = np.array(recorder.read(), dtype=np.int16)
                now = time.monotonic()

                if self._wake_debug and (now - last_debug_at) >= 2.0:
                    rms = float(np.mean(np.abs(pcm.astype(np.float32)))) if pcm.size else 0.0
                    self.logger.log(
                        "wake_debug",
                        {
                            "device_index": device_index,
                            "device_name": device_name,
                            "rms_mean_abs": round(rms, 2),
                        },
                    )
                    last_debug_at = now

                result = porcupine.process(pcm)
                if result < 0:
                    continue

                if now - last_wake_at < self._wake_cooldown_sec:
                    self.logger.log("wake_word_ignored_cooldown", {"cooldown_sec": self._wake_cooldown_sec})
                    continue
                last_wake_at = now

                self.logger.log(
                    "wake_word_detected",
                    {"word": "джарвис", "timestamp": time.time(), "sensitivity": self._porcupine_sensitivity},
                )
                self._set_status("Heard wake word")

                if self.is_busy():
                    self.speak("Подожди секунду")
                    continue

                self.speak("Слушаю")
                self._flush_audio(recorder, self._command_pre_roll_ms)
                command = self._capture_and_transcribe(
                    recorder=recorder,
                    whisper_model=whisper_model,
                    max_sec=command_max_sec,
                    silence_ms=silence_ms,
                    rms_threshold=rms_threshold,
                    beam_size=beam_size,
                    start_timeout_sec=self._command_start_timeout,
                )
                if not command and not self._stop_event.is_set():
                    self.speak("Да?")
                if command:
                    self.logger.log("stt_text", {"text": command})
                    self.on_user_text(command)

                if not self._stop_event.is_set():
                    self._set_status("Listening")

        except Exception as exc:
            self.logger.log("voice_error", {"stage": "voice_loop", "error": str(exc)})
            self.on_error(f"Ошибка voice loop: {exc}")
        finally:
            self._stop_event.set()
            try:
                if recorder is not None:
                    recorder.stop()
                    recorder.delete()
            except Exception as exc:
                self.logger.log("voice_error", {"stage": "recorder_close", "error": str(exc)})
            try:
                if porcupine is not None:
                    porcupine.delete()
            except Exception as exc:
                self.logger.log("voice_error", {"stage": "porcupine_close", "error": str(exc)})
            self.logger.log("voice_stopped", {})
            self._set_status("Idle")
            if self.on_stopped:
                self.on_stopped()

    def _capture_and_transcribe(
        self,
        recorder,
        whisper_model,
        max_sec: float,
        silence_ms: int,
        rms_threshold: float,
        beam_size: int,
        start_timeout_sec: float,
    ) -> str | None:
        max_frames = int((max_sec * 1000) / self.FRAME_MS)
        silence_limit = max(1, int(silence_ms / self.FRAME_MS))
        start_timeout_frames = max(1, int((start_timeout_sec * 1000) / self.FRAME_MS))

        frames: list[np.ndarray] = []
        speech_detected = False
        silent_frames = 0
        waited_frames = 0

        while not self._stop_event.is_set() and len(frames) < max_frames:
            chunk = np.array(recorder.read(), dtype=np.int16)
            if not speech_detected:
                waited_frames += 1

            rms = self._chunk_rms(chunk)
            is_speech = rms >= rms_threshold
            if is_speech:
                speech_detected = True
                silent_frames = 0
                frames.append(chunk)
            elif speech_detected:
                frames.append(chunk)
                silent_frames += 1
            elif waited_frames >= start_timeout_frames:
                return None

            if speech_detected and silent_frames >= silence_limit:
                break

        if not speech_detected:
            return None
        return self._transcribe_audio(whisper_model=whisper_model, frames=frames, beam_size=beam_size)

    def _transcribe_audio(self, whisper_model, frames: list[np.ndarray], beam_size: int) -> str | None:
        if self._stop_event.is_set() or not frames:
            return None
        try:
            frame_buffer = np.concatenate(frames).astype(np.int16, copy=False)
            audio_float = frame_buffer.astype(np.float32) / 32768.0
            segments, _ = whisper_model.transcribe(audio_float, language="ru", beam_size=beam_size)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return text or None
        except Exception as exc:
            self.logger.log("voice_error", {"stage": "transcribe", "error": str(exc)})
            self.on_error(f"Ошибка распознавания речи: {exc}")
            return None

    @staticmethod
    def _chunk_rms(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk.astype(np.float32)))))

    @staticmethod
    def _read_volume_from_env() -> float:
        raw = os.getenv("JARVIS_TTS_VOLUME", "80").strip()
        try:
            value = float(raw)
        except ValueError:
            return 0.8
        if value > 1.0:
            value = value / 100.0
        return max(0.0, min(1.0, value))

    def _float_env(self, key: str, default: float) -> float:
        raw = os.getenv(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            self.logger.log("voice_error", {"stage": "config", "error": f"Invalid {key}={raw}"})
            return default

    def _int_env(self, key: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
        raw = os.getenv(key)
        if raw is None:
            value = default
        else:
            try:
                value = int(float(raw))
            except ValueError:
                self.logger.log("voice_error", {"stage": "config", "error": f"Invalid {key}={raw}"})
                value = default
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return value

    def get_wake_status_line(self) -> str:
        if self._selected_device_label == "unknown":
            try:
                from pvrecorder import PvRecorder

                devices = PvRecorder.get_available_devices()
                if not devices:
                    self._selected_device_label = "нет устройств"
                else:
                    device_index = self._int_env("JARVIS_AUDIO_DEVICE_INDEX", 0)
                    if device_index == -1:
                        self._selected_device_label = "-1: auto"
                    elif 0 <= device_index < len(devices):
                        self._selected_device_label = f"{device_index}: {devices[device_index]}"
                    else:
                        self._selected_device_label = f"{device_index}: out_of_range"
            except Exception:
                pass
        return f"Wake: Porcupine (sens={self._porcupine_sensitivity:.2f}, device={self._selected_device_label})"

    def _set_status(self, status: str) -> None:
        with self._status_lock:
            if self._current_status == status:
                return
            self._current_status = status
        self.on_status(status)

    def _flush_audio(self, recorder, pre_roll_ms: int) -> None:
        frames_to_drop = max(0, int(pre_roll_ms / self.FRAME_MS))
        for _ in range(frames_to_drop):
            if self._stop_event.is_set():
                return
            recorder.read()


def sanitize_for_tts(text: str, max_len: int = 500) -> str:
    cleaned = text.replace("⚠️", "Внимание").replace("⚠", "Внимание")
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"[A-Za-z]:\\[^\s]+", "", cleaned)
    cleaned = re.sub(r"/[\w\-./]{12,}", "", cleaned)
    cleaned = re.sub(r"\{.*?\}", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\[.*?\]", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    return cleaned[:max_len]
