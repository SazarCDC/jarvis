from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
import json
from typing import Callable

import numpy as np

from jarvis_assistant.assistant_core import JarvisAssistant
from jarvis_assistant.logger import JsonlLogger
from jarvis_assistant.config import CONFIG


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

        self._tts_queue: queue.Queue[str | None] = queue.Queue()
        self._tts_thread: threading.Thread | None = None
        self._tts_stop_event = threading.Event()
        self._tts_playing_event = threading.Event()
        self._tts_backend: PiperTTS | None = None
        self._volume_0_1 = self._read_volume_from_env()
        self._followup_seconds = CONFIG.agent.followup_seconds
        self._last_dialog_ts = 0.0
        self._tts_rate = CONFIG.voice.tts_rate
        self._status_lock = threading.Lock()
        self._current_status = "Idle"

        self._porcupine_sensitivity = min(1.0, max(0.0, CONFIG.voice.porcupine_sensitivity))
        self._wake_debug = self._int_env("JARVIS_WAKE_DEBUG", 0) == 1
        self._wake_cooldown_sec = max(0.0, self._float_env("JARVIS_WAKE_COOLDOWN", 1.0))
        self._command_pre_roll_ms = max(0, self._int_env("JARVIS_COMMAND_PRE_ROLL_MS", 150))
        self._command_start_timeout = max(0.3, self._float_env("JARVIS_COMMAND_START_TIMEOUT", 6.0))
        self._wake_earcon_enabled = self._int_env("JARVIS_WAKE_EARCON", 1) == 1
        self._wake_earcon_freq = self._float_env("JARVIS_WAKE_EARCON_FREQ", 880.0)
        self._wake_earcon_ms = self._int_env("JARVIS_WAKE_EARCON_MS", 70, min_value=20, max_value=250)
        self._wake_earcon_gain = max(0.0, min(1.0, self._float_env("JARVIS_WAKE_EARCON_GAIN", 0.25)))
        self._wake_post_tts_silence_ms = self._int_env(
            "JARVIS_WAKE_POST_TTS_SILENCE_MS",
            CONFIG.voice.wake_post_tts_silence_ms,
            min_value=0,
            max_value=800,
        )
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
        self._tts_stop_event.clear()
        self._tts_backend = PiperTTS(
            logger=self.logger,
            on_error=self.on_error,
            stop_event=self._tts_stop_event,
            volume_0_1=self._volume_0_1,
            rate=self._tts_rate,
        )
        self._tts_thread = threading.Thread(target=self._tts_loop, daemon=True, name="tts-loop")
        self._tts_thread.start()

    def stop_tts(self) -> None:
        self._tts_stop_event.set()
        if self._tts_backend is not None:
            self._tts_backend.stop_playback()
        self._clear_tts_queue()
        self._enqueue_tts(None)

    def set_volume(self, volume_0_1: float) -> None:
        self._volume_0_1 = min(1.0, max(0.0, float(volume_0_1)))
        if self._tts_backend is not None:
            self._tts_backend.set_volume(self._volume_0_1)

    def speak(self, text: str) -> None:
        cleaned = sanitize_for_tts(text)
        if not cleaned or self._stop_event.is_set():
            return
        self.start_tts()
        self._enqueue_tts(cleaned)

    def _enqueue_tts(self, item: str | None) -> None:
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
        try:
            if self._tts_backend is None:
                return

            while not self._tts_stop_event.is_set():
                item = self._tts_queue.get()
                if item is None:
                    break
                if not isinstance(item, str):
                    continue

                text = item
                try:
                    self._set_status("Speaking")
                    self._tts_playing_event.set()
                    self.logger.log("tts_started", {"text": text[:120]})
                    pcm_bytes, sample_rate = self._tts_backend.synthesize_stream_raw(text)
                    if pcm_bytes is None:
                        continue
                    self._tts_backend.play(pcm_bytes=pcm_bytes, sample_rate=sample_rate)
                    if self._tts_stop_event.is_set():
                        break
                    self.logger.log("tts_finished", {"text": text[:120]})
                except Exception as exc:
                    self.logger.log("tts_error", {"error": str(exc)})
                finally:
                    self._tts_playing_event.clear()
                    self._set_status("Listening" if self.is_running() and not self._stop_event.is_set() else "Idle")
        except Exception as exc:
            self.logger.log("tts_error", {"error": str(exc), "stage": "init"})
            self.on_error(f"Ошибка синтеза речи: {exc}")
        finally:
            if self._tts_backend is not None:
                self._tts_backend.stop_playback()
            self._tts_backend = None

    def _run_loop(self) -> None:
        recorder = None
        porcupine = None
        try:
            import pvporcupine
            from faster_whisper import WhisperModel
            from pvrecorder import PvRecorder

            access_key = CONFIG.voice.picovoice_access_key.strip()
            if not access_key:
                self.on_error("Wake word недоступен: не задан JARVIS_PICOVOICE_ACCESS_KEY")
                return

            model_path = str(CONFIG.paths.porcupine_ppn)
            if not model_path or not os.path.isfile(model_path):
                self.on_error("Wake word недоступен: файл .ppn не найден")
                return

            command_max_sec = CONFIG.voice.command_max_sec
            silence_ms = CONFIG.voice.command_silence_ms
            rms_threshold = CONFIG.voice.vad_rms_threshold
            whisper_model_name = CONFIG.voice.whisper_model
            beam_size = CONFIG.voice.whisper_beam_size
            device_index = CONFIG.voice.device_index
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
                    self.logger.log("wake_word_ignored_busy", {})
                    continue

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
                if command:
                    self.logger.log("stt_text", {"text": command})
                    self._last_dialog_ts = time.time()
                    self.on_user_text(command)
                elif not self._stop_event.is_set():
                    self.logger.log("stt_empty", {})

                if (self.assistant.awaiting_user or (time.time() - self._last_dialog_ts) <= self._followup_seconds) and not self._stop_event.is_set() and not self.is_busy():
                    followup = self._capture_and_transcribe(
                        recorder=recorder,
                        whisper_model=whisper_model,
                        max_sec=min(6.0, command_max_sec),
                        silence_ms=silence_ms,
                        rms_threshold=rms_threshold,
                        beam_size=beam_size,
                        start_timeout_sec=2.0,
                    )
                    if followup:
                        self.logger.log("stt_followup_text", {"text": followup})
                        self._last_dialog_ts = time.time()
                        self.on_user_text(followup)

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
                    device_index = CONFIG.voice.device_index
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

    def wait_for_tts_idle(self, timeout_sec: float = 2.0) -> bool:
        if timeout_sec <= 0:
            return not self._tts_playing_event.is_set()
        deadline = time.monotonic() + timeout_sec
        while self._tts_playing_event.is_set() and time.monotonic() < deadline:
            if self._stop_event.is_set():
                break
            time.sleep(0.02)
        return not self._tts_playing_event.is_set()

    def _play_wake_earcon(self) -> None:
        try:
            import sounddevice as sd

            sample_rate = 22050
            duration_sec = self._wake_earcon_ms / 1000.0
            samples = max(1, int(sample_rate * duration_sec))
            timeline = np.arange(samples, dtype=np.float32) / float(sample_rate)
            wave = np.sin(2.0 * np.pi * float(self._wake_earcon_freq) * timeline).astype(np.float32)

            fade_samples = max(1, int(sample_rate * 0.01))
            fade_samples = min(fade_samples, max(1, samples // 2))
            if fade_samples > 0:
                fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                wave[:fade_samples] *= fade
                wave[-fade_samples:] *= fade[::-1]

            wave *= float(self._wake_earcon_gain)
            sd.play(wave, sample_rate, blocking=True)
            self.logger.log(
                "wake_earcon_played",
                {
                    "freq": self._wake_earcon_freq,
                    "ms": self._wake_earcon_ms,
                    "gain": self._wake_earcon_gain,
                },
            )
        except Exception as exc:
            self.logger.log("voice_error", {"stage": "wake_earcon", "error": str(exc)})

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


class PiperTTS:
    def __init__(
        self,
        logger: JsonlLogger,
        on_error: Callable[[str], None],
        stop_event: threading.Event,
        volume_0_1: float,
        rate: float,
    ) -> None:
        self.logger = logger
        self.on_error = on_error
        self.stop_event = stop_event
        self._volume_0_1 = volume_0_1
        self._rate = rate
        self._playback_lock = threading.Lock()
        self._current_stream = None
        self._piper_process: subprocess.Popen | None = None
        self._process_lock = threading.Lock()

        self._backend = "piper"
        self._model_path = str(CONFIG.paths.piper_model)
        self._piper_exe_path = str(CONFIG.paths.piper_exe)
        self._available = False
        self._default_sample_rate = 22050
        self._default_sample_rate = self._resolve_sample_rate_from_model_config()
        self._init_backend()

    def _init_backend(self) -> None:
        if self._backend != "piper":
            self.on_error(f"TTS недоступен: неподдерживаемый backend {self._backend}")
            return
        if not self._model_path:
            self.on_error("TTS недоступен: не задан JARVIS_PIPER_MODEL_PATH")
            return
        if not os.path.isfile(self._model_path):
            self.on_error(f"TTS недоступен: модель Piper не найдена: {self._model_path}")
            return
        if self._piper_exe_path and os.path.isfile(self._piper_exe_path):
            self._available = True
            self.logger.log(
                "tts_backend_selected",
                {"backend": "piper_subprocess_raw", "exe": self._piper_exe_path, "sample_rate": self._default_sample_rate},
            )
            return

        self.on_error("TTS недоступен: JARVIS_PIPER_EXE_PATH обязателен и должен указывать на piper.exe")

    def _resolve_sample_rate_from_model_config(self) -> int:
        if not self._model_path:
            return 22050
        candidates = [f"{self._model_path}.json"]
        if self._model_path.endswith(".onnx"):
            candidates.append(self._model_path.replace(".onnx", ".onnx.json"))
        for config_path in candidates:
            if not os.path.isfile(config_path):
                continue
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sample_rate = data.get("audio", {}).get("sample_rate")
                if isinstance(sample_rate, int) and sample_rate > 1000:
                    return sample_rate
            except Exception as exc:
                self.logger.log("tts_error", {"stage": "piper_sample_rate", "error": str(exc), "path": config_path})
        return 22050

    def _register_process(self, process: subprocess.Popen | None) -> None:
        with self._process_lock:
            self._piper_process = process

    def _terminate_piper_process(self) -> None:
        process = None
        with self._process_lock:
            if self._piper_process is not None and self._piper_process.poll() is None:
                process = self._piper_process
            self._piper_process = None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.5)
        except Exception as exc:
            self.logger.log("tts_error", {"stage": "piper_terminate", "error": str(exc)})
            try:
                process.kill()
            except Exception:
                pass

    def set_volume(self, volume_0_1: float) -> None:
        self._volume_0_1 = min(1.0, max(0.0, float(volume_0_1)))

    def synthesize_stream_raw(self, text: str) -> tuple[bytes | None, int]:
        if not self._available or self.stop_event.is_set():
            return None, self._default_sample_rate
        cmd = [self._piper_exe_path, "--model", self._model_path, "--output_raw"]
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._register_process(process)
            payload = f"{text}\n".encode("utf-8")
            stdout_data, stderr_data = process.communicate(input=payload)
            if self.stop_event.is_set():
                return None, self._default_sample_rate
            if process.returncode != 0:
                stderr_text = stderr_data.decode("utf-8", errors="replace").strip()
                raise RuntimeError(stderr_text or "piper returned non-zero exit code")
            return stdout_data, self._default_sample_rate
        except Exception as exc:
            self.logger.log("tts_error", {"stage": "piper_subprocess_synthesize", "error": str(exc)})
            self.on_error(f"Ошибка Piper TTS: {exc}")
            return None, self._default_sample_rate
        finally:
            self._register_process(None)

    def play(self, pcm_bytes: bytes, sample_rate: int) -> None:
        if self.stop_event.is_set() or not pcm_bytes:
            return
        try:
            import sounddevice as sd

            audio = np.frombuffer(pcm_bytes, dtype=np.int16)
            if audio.size == 0:
                return
            scaled = np.clip(audio.astype(np.float32) * self._volume_0_1, -32768, 32767).astype(np.int16)
            chunk_size = max(1, int(sample_rate * 0.1))
            with self._playback_lock:
                self._current_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16")
                self._current_stream.start()
                start = 0
                while start < scaled.size and not self.stop_event.is_set():
                    self._current_stream.write(scaled[start : start + chunk_size])
                    start += chunk_size
                self._current_stream.stop()
                self._current_stream.close()
                self._current_stream = None
        except Exception as exc:
            self.logger.log("tts_error", {"stage": "playback", "error": str(exc)})
            self.on_error(f"Ошибка воспроизведения TTS: {exc}")

    def stop_playback(self) -> None:
        self.logger.log("tts_stopped", {})
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

        self._terminate_piper_process()

        with self._playback_lock:
            if self._current_stream is not None:
                try:
                    self._current_stream.abort(ignore_errors=True)
                except Exception:
                    pass
                try:
                    self._current_stream.close(ignore_errors=True)
                except Exception:
                    pass
                self._current_stream = None
