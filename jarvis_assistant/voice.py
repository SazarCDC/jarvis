from __future__ import annotations

import os
import queue
import re
import threading
import time
from typing import Callable, Protocol

import numpy as np

from jarvis_assistant.assistant_core import JarvisAssistant
from jarvis_assistant.logger import JsonlLogger


class VadBackend(Protocol):
    def is_speech(self, pcm16: bytes, sample_rate: int) -> bool: ...


class LiteVadBackend:
    def __init__(self, rms_threshold: float) -> None:
        self.rms_threshold = max(1.0, float(rms_threshold))

    def is_speech(self, pcm16: bytes, sample_rate: int) -> bool:
        del sample_rate
        audio = np.frombuffer(pcm16, dtype=np.int16)
        if audio.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
        return rms >= self.rms_threshold


class SileroVadBackend:
    def __init__(self, threshold: float) -> None:
        self.threshold = min(1.0, max(0.0, float(threshold)))
        import torch

        model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        model.eval()

        self._torch = torch
        self._model = model

    def is_speech(self, pcm16: bytes, sample_rate: int) -> bool:
        if sample_rate != 16000:
            return False
        audio = np.frombuffer(pcm16, dtype=np.int16)
        if audio.size == 0:
            return False

        tensor = self._torch.from_numpy(audio.astype(np.float32) / 32768.0)
        with self._torch.no_grad():
            probability = float(self._model(tensor, sample_rate).item())
        return probability >= self.threshold


class VoiceController:
    SAMPLE_RATE = 16000
    CHANNELS = 1

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
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=300)
        self._volume_0_1 = self._read_volume_from_env()
        self._vad_backend: VadBackend | None = None

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._clear_audio_queue()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="voice-loop")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._stop_tts()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def set_volume(self, volume_0_1: float) -> None:
        self._volume_0_1 = min(1.0, max(0.0, float(volume_0_1)))

    def speak(self, text: str) -> None:
        cleaned = sanitize_for_tts(text)
        if not cleaned:
            return
        if self._stop_event.is_set():
            return

        self.on_status("Speaking")
        self.logger.log("tts_started", {"text": cleaned})
        try:
            with self._tts_lock:
                if self._stop_event.is_set():
                    return
                engine = self._init_tts_engine()
                if engine is None:
                    return

                engine.stop()
                engine.setProperty("volume", self._volume_0_1)
                engine.say(cleaned)
                run_error: Exception | None = None

                def _run_and_wait() -> None:
                    nonlocal run_error
                    try:
                        engine.runAndWait()
                    except Exception as exc:  # pragma: no cover - depends on TTS backend
                        run_error = exc

                worker = threading.Thread(target=_run_and_wait, daemon=True, name="tts-runner")
                worker.start()
                worker.join(timeout=8)

                if worker.is_alive():
                    self.logger.log("voice_error", {"stage": "tts", "error": "runAndWait timeout"})
                    engine.stop()
                    self._tts_engine = None
                    return
                if run_error is not None:
                    self.logger.log("voice_error", {"stage": "tts", "error": str(run_error)})
                    engine.stop()
                    self._tts_engine = None
                    return
        except Exception as exc:
            self.logger.log("voice_error", {"stage": "tts", "error": str(exc)})
            self.on_error(f"Ошибка синтеза речи: {exc}")
        finally:
            self.logger.log("tts_finished", {"text": cleaned})
            self.on_status("Listening" if self.is_running() and not self._stop_event.is_set() else "Idle")

    def _run_loop(self) -> None:
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
            from openwakeword.model import Model

            wake_backend = (os.getenv("JARVIS_WAKE_BACKEND", "onnx").strip().lower() or "onnx")
            wake_model_name = os.getenv("JARVIS_WAKE_MODEL", "hey_jarvis").strip() or "hey_jarvis"
            if wake_backend != "onnx":
                self.logger.log(
                    "voice_error",
                    {"stage": "wake_init", "error": f"Unsupported wake backend: {wake_backend}"},
                )
                self.on_error("Wake word недоступен: требуется onnxruntime")
                return
            try:
                wake_model = Model(
                    inference_framework="onnx",
                    wakeword_models=[wake_model_name],
                )
            except Exception as exc:
                self.logger.log("voice_error", {"stage": "wake_init", "error": str(exc)})
                self.on_error("Wake word недоступен: требуется onnxruntime")
                return
            self.logger.log(
                "wake_backend_selected",
                {"backend": "openwakeword", "framework": wake_backend, "model": wake_model_name},
            )

            wake_threshold = self._float_env("JARVIS_WAKE_THRESHOLD", 0.65)
            wake_cooldown = self._float_env("JARVIS_WAKE_COOLDOWN", 2.0)
            start_timeout = self._float_env("JARVIS_COMMAND_START_TIMEOUT", 3.0)
            silence_ms = self._int_env("JARVIS_COMMAND_SILENCE_MS", 1000, min_value=200, max_value=3000)
            max_sec = self._float_env("JARVIS_COMMAND_MAX_SEC", 10.0)
            whisper_model_name = os.getenv("JARVIS_WHISPER_MODEL", "small").strip() or "small"
            beam_size = self._int_env("JARVIS_WHISPER_BEAM_SIZE", 1, min_value=1, max_value=3)

            vad = self._resolve_vad_backend()
            whisper_model = WhisperModel(whisper_model_name, device="cpu", compute_type="int8")
            device = self._resolve_audio_device(sd)

            def audio_callback(indata, frames, callback_time, status) -> None:
                del frames, callback_time
                if status:
                    self.logger.log("voice_error", {"stage": "audio_callback", "error": str(status)})
                chunk = np.copy(indata[:, 0]).astype(np.int16)
                try:
                    self._audio_queue.put_nowait(chunk)
                except queue.Full:
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._audio_queue.put_nowait(chunk)

            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype="int16",
                callback=audio_callback,
                blocksize=320,
                device=device,
            ):
                self.logger.log("voice_started", {})
                self.on_status("Listening")
                consecutive_hits = 0
                cooldown_until = 0.0
                while not self._stop_event.is_set():
                    chunk = self._next_chunk(timeout=0.2)
                    if chunk is None:
                        continue

                    if time.monotonic() < cooldown_until:
                        continue

                    score, score_model_name = self._wake_score(wake_model, chunk, wake_model_name)
                    if score >= wake_threshold:
                        consecutive_hits += 1
                    else:
                        consecutive_hits = 0

                    if consecutive_hits < 3:
                        continue

                    consecutive_hits = 0
                    cooldown_until = time.monotonic() + wake_cooldown
                    self.logger.log(
                        "wake_word_detected",
                        {
                            "score": round(score, 4),
                            "threshold": wake_threshold,
                            "model": score_model_name,
                        },
                    )
                    self.on_status("Heard wake word")

                    if self.is_busy():
                        self.speak("Подожди секунду")
                        continue

                    self.speak("Слушаю")
                    command = self._capture_and_transcribe(
                        vad=vad,
                        whisper_model=whisper_model,
                        start_timeout=start_timeout,
                        silence_ms=silence_ms,
                        max_sec=max_sec,
                        beam_size=beam_size,
                    )
                    if not command and not self._stop_event.is_set():
                        self.speak("Да?")
                        command = self._capture_and_transcribe(
                            vad=vad,
                            whisper_model=whisper_model,
                            start_timeout=start_timeout,
                            silence_ms=silence_ms,
                            max_sec=max_sec,
                            beam_size=beam_size,
                        )

                    if command:
                        self.logger.log("stt_text", {"text": command})
                        self.on_user_text(command)
                    self.on_status("Listening")

        except Exception as exc:
            self.logger.log("voice_error", {"stage": "voice_loop", "error": str(exc)})
            self.on_error(f"Ошибка voice loop: {exc}")
        finally:
            self._stop_event.set()
            self.logger.log("voice_stopped", {})
            self.on_status("Idle")
            if self.on_stopped:
                self.on_stopped()

    def _resolve_vad_backend(self) -> VadBackend:
        if self._vad_backend is not None:
            return self._vad_backend

        requested = os.getenv("JARVIS_VAD_BACKEND", "silero").strip().lower()
        silero_threshold = self._float_env("JARVIS_SILERO_VAD_THRESHOLD", 0.5)
        lite_threshold = self._float_env("JARVIS_VAD_RMS_THRESHOLD", 700.0)

        if requested == "lite":
            backend = LiteVadBackend(rms_threshold=lite_threshold)
            self.logger.log("vad_backend_selected", {"backend": "lite", "rms_threshold": lite_threshold})
            self._vad_backend = backend
            return backend

        try:
            backend = SileroVadBackend(threshold=silero_threshold)
            self.logger.log(
                "vad_backend_selected",
                {"backend": "silero", "threshold": silero_threshold},
            )
            self._vad_backend = backend
            return backend
        except Exception as exc:
            self.logger.log("voice_error", {"stage": "vad_init", "error": str(exc)})
            self.on_error("Silero VAD недоступен, переключаюсь на VAD-lite")
            backend = LiteVadBackend(rms_threshold=lite_threshold)
            self.logger.log(
                "vad_backend_selected",
                {
                    "backend": "lite",
                    "fallback": "silero_init_failed",
                    "rms_threshold": lite_threshold,
                },
            )
            self._vad_backend = backend
            return backend

    def _capture_and_transcribe(
        self,
        vad: VadBackend,
        whisper_model,
        start_timeout: float,
        silence_ms: int,
        max_sec: float,
        beam_size: int,
    ) -> str | None:
        frame_samples = 320
        frame_bytes = frame_samples * 2
        start_deadline = time.monotonic() + start_timeout
        silence_frames_limit = max(1, int(silence_ms / 20))

        collected = np.empty(0, dtype=np.int16)
        speech_started = False
        silence_frames = 0
        speech_start_ts = 0.0
        speech_frames: list[np.ndarray] = []

        while not self._stop_event.is_set():
            timeout = 0.3 if speech_started else max(0.1, start_deadline - time.monotonic())
            chunk = self._next_chunk(timeout=timeout)
            if chunk is None:
                if not speech_started and time.monotonic() > start_deadline:
                    return None
                continue

            collected = np.concatenate((collected, chunk))
            while collected.size >= frame_samples:
                frame = collected[:frame_samples]
                collected = collected[frame_samples:]
                if len(frame.tobytes()) != frame_bytes:
                    continue
                speech = vad.is_speech(frame.tobytes(), self.SAMPLE_RATE)
                if speech:
                    if not speech_started:
                        speech_started = True
                        speech_start_ts = time.monotonic()
                        self.logger.log("vad_start", {})
                    silence_frames = 0
                elif speech_started:
                    silence_frames += 1

                if speech_started:
                    speech_frames.append(frame.copy())

                if speech_started and silence_frames >= silence_frames_limit:
                    duration_ms = int((time.monotonic() - speech_start_ts) * 1000)
                    self.logger.log("vad_end", {"duration_ms": duration_ms})
                    return self._transcribe_audio(whisper_model, frames=speech_frames, beam_size=beam_size)

                if speech_started and (time.monotonic() - speech_start_ts) >= max_sec:
                    duration_ms = int((time.monotonic() - speech_start_ts) * 1000)
                    self.logger.log("vad_end", {"duration_ms": duration_ms})
                    return self._transcribe_audio(whisper_model, frames=speech_frames, beam_size=beam_size)

        return None

    def _transcribe_audio(self, whisper_model, frames: list[np.ndarray], beam_size: int) -> str | None:
        if self._stop_event.is_set() or not frames:
            return None
        try:
            frame_buffer = np.concatenate(frames).astype(np.int16, copy=False)
            audio_float = frame_buffer.astype(np.float32) / 32768.0
            segments, _ = whisper_model.transcribe(audio_float, language="ru", beam_size=beam_size)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if text:
                return text
            self.speak("Не расслышал")
            return None
        except Exception as exc:
            self.logger.log("voice_error", {"stage": "transcribe", "error": str(exc)})
            self.on_error(f"Ошибка распознавания речи: {exc}")
            return None

    def _wake_score(self, wake_model, chunk: np.ndarray, fallback_name: str) -> tuple[float, str]:
        prediction = wake_model.predict(chunk)
        if not isinstance(prediction, dict) or not prediction:
            return 0.0, fallback_name

        if "hey_jarvis" in prediction:
            return float(prediction["hey_jarvis"]), "hey_jarvis"

        best_name, best_score = max(prediction.items(), key=lambda item: float(item[1]))
        return float(best_score), str(best_name)

    def _next_chunk(self, timeout: float) -> np.ndarray | None:
        if self._stop_event.is_set():
            return None
        try:
            return self._audio_queue.get(timeout=max(0.01, timeout))
        except queue.Empty:
            return None

    def _clear_audio_queue(self) -> None:
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def _resolve_audio_device(self, sd):
        value = os.getenv("JARVIS_AUDIO_DEVICE", "").strip()
        if not value:
            return None
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)

        try:
            devices = sd.query_devices()
            for idx, item in enumerate(devices):
                name = str(item.get("name", ""))
                max_input = int(item.get("max_input_channels", 0))
                if max_input > 0 and value.lower() in name.lower():
                    return idx
        except Exception as exc:
            self.logger.log("voice_error", {"stage": "audio_device", "error": str(exc)})
        return None

    def _init_tts_engine(self):
        if self._tts_engine is not None:
            return self._tts_engine
        try:
            import pyttsx3

            self._tts_engine = pyttsx3.init()
            return self._tts_engine
        except Exception as exc:
            self.logger.log("voice_error", {"stage": "tts_init", "error": str(exc)})
            self._tts_engine = None
            return None

    def _stop_tts(self) -> None:
        with self._tts_lock:
            if self._tts_engine is not None:
                self._tts_engine.stop()

    @staticmethod
    def _read_volume_from_env() -> float:
        raw = os.getenv("JARVIS_TTS_VOLUME", "0.8").strip()
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
