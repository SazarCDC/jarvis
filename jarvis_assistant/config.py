from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PathsConfig:
    memory_dir: Path
    log_dir: Path
    tools_dir: Path
    models_dir: Path
    piper_exe: Path
    piper_model: Path
    porcupine_ppn: Path


@dataclass(slots=True)
class LLMConfig:
    ollama_host: str
    ollama_model: str


@dataclass(slots=True)
class AgentConfig:
    max_steps: int
    max_history: int
    followup_seconds: int
    summary_trigger: int


@dataclass(slots=True)
class VoiceConfig:
    device_index: int
    porcupine_sensitivity: float
    vad_rms_threshold: float
    command_max_sec: float
    command_silence_ms: int
    command_start_timeout: float
    wake_post_tts_silence_ms: int
    tts_volume: float
    tts_rate: float
    whisper_model: str
    whisper_beam_size: int
    picovoice_access_key: str


@dataclass(slots=True)
class WebConfig:
    enabled: bool
    timeout_sec: int
    max_results: int
    region: str
    fetch_max_chars: int
    extract_max_chars: int


@dataclass(slots=True)
class RuntimeConfig:
    command_timeout_sec: int


@dataclass(slots=True)
class Config:
    project_root: Path
    paths: PathsConfig
    llm: LLMConfig
    agent: AgentConfig
    voice: VoiceConfig
    web: WebConfig
    runtime: RuntimeConfig

    @property
    def ollama_host(self) -> str:
        return self.llm.ollama_host

    @property
    def ollama_model(self) -> str:
        return self.llm.ollama_model


class ConfigLoader:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent

    def load(self) -> Config:
        settings_path = self.project_root / "settings.json"
        if not settings_path.exists():
            settings_path = self.project_root / "settings.example.json"
        data = self._read_json(settings_path)

        paths = data.get("paths", {})
        path_cfg = PathsConfig(
            memory_dir=self._path_value(paths, "memory_dir", "jarvis_assistant/memory", "JARVIS_MEMORY_DIR"),
            log_dir=self._path_value(paths, "log_dir", "logs", "JARVIS_LOG_DIR"),
            tools_dir=self._path_value(paths, "tools_dir", "tools", "JARVIS_TOOLS_DIR"),
            models_dir=self._path_value(paths, "models_dir", "models", "JARVIS_MODELS_DIR"),
            piper_exe=self._path_value(paths, "piper_exe", "tools/piper/piper.exe", "JARVIS_PIPER_EXE_PATH"),
            piper_model=self._path_value(paths, "piper_model", "models/ru_RU-denis-medium.onnx", "JARVIS_PIPER_MODEL_PATH"),
            porcupine_ppn=self._path_value(paths, "porcupine_ppn", "models/jarvis.ppn", "JARVIS_PICOVOICE_MODEL_PATH"),
        )

        llm = data.get("llm", {})
        llm_cfg = LLMConfig(
            ollama_host=self._env_or(llm, "ollama_host", "http://127.0.0.1:11434", "OLLAMA_HOST"),
            ollama_model=self._env_or(llm, "ollama_model", "qwen2.5:7b-instruct", "OLLAMA_MODEL"),
        )

        agent = data.get("agent", {})
        agent_cfg = AgentConfig(
            max_steps=self._int(self._env_or(agent, "max_steps", 12, "JARVIS_AGENT_MAX_STEPS"), 12),
            max_history=self._int(self._env_or(agent, "max_history", 20, "JARVIS_MAX_HISTORY"), 20),
            followup_seconds=self._int(self._env_or(agent, "followup_seconds", 8, "JARVIS_FOLLOWUP_SECONDS"), 8),
            summary_trigger=self._int(self._env_or(agent, "summary_trigger", 30, "JARVIS_SUMMARY_TRIGGER"), 30),
        )

        voice = data.get("voice", {})
        voice_cfg = VoiceConfig(
            device_index=self._int(self._env_or(voice, "device_index", 0, "JARVIS_AUDIO_DEVICE_INDEX"), 0),
            porcupine_sensitivity=self._float(self._env_or(voice, "porcupine_sensitivity", 0.7, "JARVIS_PORCUPINE_SENSITIVITY"), 0.7),
            vad_rms_threshold=self._float(self._env_or(voice, "vad_rms_threshold", 300, "JARVIS_VAD_RMS_THRESHOLD"), 300),
            command_max_sec=self._float(self._env_or(voice, "command_max_sec", 15, "JARVIS_COMMAND_MAX_SEC"), 15),
            command_silence_ms=self._int(self._env_or(voice, "command_silence_ms", 1400, "JARVIS_COMMAND_SILENCE_MS"), 1400),
            command_start_timeout=self._float(self._env_or(voice, "command_start_timeout", 6, "JARVIS_COMMAND_START_TIMEOUT"), 6),
            wake_post_tts_silence_ms=self._int(self._env_or(voice, "wake_post_tts_silence_ms", 120, "JARVIS_WAKE_POST_TTS_SILENCE_MS"), 120),
            tts_volume=self._float(self._env_or(voice, "tts_volume", 80, "JARVIS_TTS_VOLUME"), 80),
            tts_rate=self._float(self._env_or(voice, "tts_rate", 1.0, "JARVIS_TTS_RATE"), 1.0),
            whisper_model=str(self._env_or(voice, "whisper_model", "small", "JARVIS_WHISPER_MODEL")),
            whisper_beam_size=self._int(self._env_or(voice, "whisper_beam_size", 1, "JARVIS_WHISPER_BEAM_SIZE"), 1),
            picovoice_access_key=str(self._env_or(voice, "picovoice_access_key", "", "JARVIS_PICOVOICE_ACCESS_KEY")),
        )

        web = data.get("web", {})
        web_cfg = WebConfig(
            enabled=bool(self._env_or(web, "enabled", True, "JARVIS_WEB_ENABLED")),
            timeout_sec=self._int(self._env_or(web, "timeout_sec", 15, "JARVIS_WEB_TIMEOUT_SEC"), 15),
            max_results=self._int(self._env_or(web, "max_results", 5, "JARVIS_WEB_MAX_RESULTS"), 5),
            region=str(self._env_or(web, "region", "ru-ru", "JARVIS_WEB_REGION")),
            fetch_max_chars=self._int(self._env_or(web, "fetch_max_chars", 60000, "JARVIS_WEB_FETCH_MAX_CHARS"), 60000),
            extract_max_chars=self._int(self._env_or(web, "extract_max_chars", 12000, "JARVIS_WEB_EXTRACT_MAX_CHARS"), 12000),
        )

        runtime = data.get("runtime", {})
        runtime_cfg = RuntimeConfig(
            command_timeout_sec=self._int(self._env_or(runtime, "command_timeout_sec", 120, "JARVIS_COMMAND_TIMEOUT"), 120)
        )

        return Config(
            project_root=self.project_root,
            paths=path_cfg,
            llm=llm_cfg,
            agent=agent_cfg,
            voice=voice_cfg,
            web=web_cfg,
            runtime=runtime_cfg,
        )

    def _path_value(self, section: dict[str, Any], key: str, default: str, env_name: str) -> Path:
        value = str(self._env_or(section, key, default, env_name))
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _env_or(section: dict[str, Any], key: str, default: Any, env_name: str) -> Any:
        return os.getenv(env_name, section.get(key, default))

    @staticmethod
    def _int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default


CONFIG = ConfigLoader().load()
