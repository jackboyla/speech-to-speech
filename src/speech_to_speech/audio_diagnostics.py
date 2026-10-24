"""Allowlisted support information; no capture, model construction, or provider requests."""

from __future__ import annotations

import json
import logging
import os
import platform
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from speech_to_speech import __version__

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig
    from speech_to_speech.cli import Command
    from speech_to_speech.config import ResolvedConfig
    from speech_to_speech.s2s_pipeline import ParsedArguments


def _audio_devices(
    client: RealtimeAudioClientConfig, devices: tuple[int | None, int | None] | None = None
) -> dict[str, Any]:
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        return {"status": "unavailable: sounddevice or PortAudio could not be loaded"}

    result: dict[str, Any] = {}
    input_device, output_device = devices or (client.input_device, client.output_device)
    for kind, selected in (("input", input_device), ("output", output_device)):
        try:
            info = sd.query_devices(selected, kind)
            result[kind] = {
                key: info[key]
                for key in (
                    "index",
                    "name",
                    "default_samplerate",
                    "max_input_channels",
                    "max_output_channels",
                    "default_low_input_latency",
                    "default_high_input_latency",
                    "default_low_output_latency",
                    "default_high_output_latency",
                )
            }
            result[kind]["host_api"] = sd.query_hostapis(info["hostapi"])["name"]
        except (sd.PortAudioError, ValueError):
            result[kind] = {"status": "unavailable: selected device could not be queried"}
    return result


def build_audio_report(
    command: Command,
    *,
    args: ParsedArguments | None = None,
    client: RealtimeAudioClientConfig | None = None,
    streams: Sequence[Any] = (),
    started_streams: Sequence[Any] = (),
    trigger: str | None = None,
) -> dict[str, Any]:
    """Collect only selected settings and metadata, never an argument/config dump."""
    versions = {"speech-to-speech source": __version__}
    for package in (
        "speech-to-speech",
        "sounddevice",
        "numpy",
        "torch",
        "transformers",
        "onnxruntime",
        "mlx",
        "mlx-lm",
        "mlx-audio",
        "nano-parakeet",
        "faster-qwen3-tts",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    system = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    if platform.system() == "Linux":
        try:
            system["linux_distribution"] = platform.freedesktop_os_release().get("PRETTY_NAME", "unknown")
        except OSError:
            system["linux_distribution"] = "unavailable"

    configured: dict[str, Any] = {}
    if args is not None:
        configured["pipeline_sample_rate_hz"] = args.vad_handler_kwargs.sample_rate
        configured["backends"] = {
            "vad": {
                "backend": args.vad_handler_kwargs.vad,
                "firered_use_gpu": args.vad_handler_kwargs.vad_firered_use_gpu,
            }
        }
        for kind in ("stt", "llm", "tts"):
            selection = getattr(args, f"{kind}_backend")
            configured["backends"][kind] = {
                "backend": selection.name,
                "device": selection.config.get("device", "not exposed by backend configuration"),
                **{
                    key: selection.config[key]
                    for key in ("model_name", "model", "sample_rate", "audio_sample_rate", "blocksize", "chunk_size")
                    if key in selection.config
                },
            }
            if "backend" in selection.config:
                configured["backends"][kind]["engine"] = selection.config["backend"]
        module = args.module_kwargs
        if module.diarization_model_name:
            configured["backends"]["diarization"] = {
                "model_name": module.diarization_model_name,
                "device": module.device or module.diarization_device,
            }
    else:
        configured["backends"] = "unknown: talk cannot inspect the remote server's configuration"
    if client is not None:
        configured["audio_client"] = {
            "input_device": client.input_device,
            "output_device": client.output_device,
            "send_rate_hz": client.send_rate,
            "recv_rate_hz": client.recv_rate,
            "callback_block_size_frames": client.chunk_size,
            "playback_buffer_ms": client.playback_buffer_ms,
            "channels": 1,
            "dtype": "int16",
            "block_mic_during_playback": client.block_mic_during_playback,
        }
    stream_info = {
        kind: {
            "device": stream.device,
            "sample_rate_hz": stream.samplerate,
            "callback_block_size_frames": stream.blocksize,
            "latency_estimate_s": stream.latency,
            "started": stream in started_streams,
        }
        for kind, stream in zip(("input", "output"), streams)
    }
    report = {
        "report": "speech-to-speech audio diagnostics",
        "command": command,
        "versions": {"python": platform.python_version(), **versions},
        "system": system,
        "configured": configured,
        "observed": {
            "audio_devices": _audio_devices(
                client,
                (
                    stream_info.get("input", {}).get("device", client.input_device),
                    stream_info.get("output", {}).get("device", client.output_device),
                ),
            )
            if client is not None
            else "not queried: serve does not own microphone/speaker devices",
            "streams": stream_info
            or "not opened: rates, block sizes and latencies above are settings or device defaults",
            "backend_devices": "not observed: configured devices may differ after automatic selection or fallback",
            "microphone_overflows": (
                "at least one input overflow observed; cumulative count unavailable"
                if trigger == "microphone input overflow"
                else "unavailable: the client retains no overflow counter"
            ),
        },
    }
    if trigger is not None:
        report["trigger"] = trigger
    return report


def print_audio_report(
    command: Command, *, args: ParsedArguments | None = None, client: RealtimeAudioClientConfig | None = None
) -> None:
    print(json.dumps(build_audio_report(command, args=args, client=client), indent=2, ensure_ascii=True))


def log_audio_report(
    *,
    client: RealtimeAudioClientConfig,
    args: ParsedArguments | None,
    streams: Sequence[Any],
    started_streams: Sequence[Any],
    trigger: str,
    level: int = logging.WARNING,
) -> None:
    """Diagnostics must not replace an audio failure or stop a working session."""
    logger = logging.getLogger(__name__)
    try:
        report = build_audio_report(
            "local" if args is not None else "talk",
            args=args,
            client=client,
            streams=streams,
            started_streams=started_streams,
            trigger=trigger,
        )
        logger.log(
            level,
            "Audio diagnostic report (copy with the warning/error for support):\n%s",
            json.dumps(report, indent=2),
        )
    except Exception:
        logger.log(level, "Audio diagnostic report unavailable")


def print_configured_audio_report(command: Command, config: ResolvedConfig) -> None:
    from speech_to_speech.configured_runtime import _adapt, build_configured_client

    args = None if command == "talk" else next(iter(_adapt(config).values()))
    client = build_configured_client(config, local=command == "local", load_tools=False) if command != "serve" else None
    print_audio_report(command, args=args, client=client)
