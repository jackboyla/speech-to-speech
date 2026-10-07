from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocalAudioArguments:
    local_audio_tool_module: Optional[str] = field(
        default=None,
        metadata={
            "help": "Importable module defining TOOLS and async execute_tool(name, arguments).",
            "aliases": ["--tool-module"],
        },
    )
    local_audio_input_device: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sounddevice input device index used by the local command."},
    )
    local_audio_output_device: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sounddevice output device index used by the local command."},
    )
    local_audio_chunk_size: int = field(
        default=1024,
        metadata={"help": "Microphone and speaker callback block size in samples. Default is 1024."},
    )
    local_audio_playback_buffer_ms: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Audio to buffer before local playback starts, in milliseconds. "
                "Defaults to 196 for OpenAI-compatible TTS and 0 otherwise."
            ),
            "aliases": ["--playback-buffer-ms"],
        },
    )
    local_audio_block_mic_during_playback: bool = field(
        default=False,
        metadata={
            "help": "Pause local microphone capture while audio is playing. Disabled by default so barge-in works."
        },
    )
    local_audio_wake_word: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Send microphone audio only after this wake word: alexa, hey_jarvis, hey_mycroft, hey_rhasspy, "
                "okay_nabu, or the path to an openWakeWord .tflite model. Needs the wakeword extra."
            ),
            "aliases": ["--wake-word"],
        },
    )
    local_audio_wake_word_threshold: float = field(
        default=0.5,
        metadata={
            "help": "Detection score from 0 to 1 needed to wake. Raise it if the client wakes by mistake.",
            "aliases": ["--wake-word-threshold"],
        },
    )
    local_audio_wake_word_timeout_s: float = field(
        default=8.0,
        metadata={
            "help": "Seconds of quiet, after the reply finishes playing, before the wake word is needed again.",
            "aliases": ["--wake-word-timeout"],
        },
    )
    local_audio_print_json: bool = field(
        default=False,
        metadata={"help": "Print raw Realtime events received by the packaged local audio client."},
    )
