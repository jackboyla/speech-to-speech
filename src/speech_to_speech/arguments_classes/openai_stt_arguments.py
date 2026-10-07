from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenAICompatibleSTTHandlerArguments:
    """Connection settings for an OpenAI-compatible STT server."""

    openai_stt_base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "Base URL of the OpenAI-compatible server, including /v1."},
    )
    openai_stt_api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional bearer token. For https://api.openai.com/v1 only, "
            "OPENAI_API_KEY is used when this flag is unset."
        },
    )
    openai_stt_model: Optional[str] = field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        metadata={
            "help": "Optional model identifier sent to POST /audio/transcriptions. "
            "May be omitted when the server selects a model from the language."
        },
    )
    openai_stt_language: Optional[str] = field(
        default=None,
        metadata={"help": "Optional ISO language hint sent with each transcription request."},
    )
    openai_stt_response_format: str = field(
        default="json",
        metadata={"help": "Transcription response format. The adapter supports json, text and verbose_json."},
    )
    openai_stt_timeout: float = field(
        default=60.0,
        metadata={"help": "HTTP request timeout in seconds."},
    )
    openai_stt_window_seconds: float = field(
        default=0.0,
        metadata={"help": "Maximum audio seconds per HTTP request. 0 disables windowing. Try 30 for Qwen3-ASR/vLLM."},
    )
    openai_stt_overlap_seconds: float = field(
        default=2.0,
        metadata={"help": "Audio overlap between STT windows; must be less than the window. Default: 2 seconds."},
    )
    openai_stt_aligner_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional local word-timing model, e.g. Qwen/Qwen3-ForcedAligner-0.6B-hf, when the HTTP backend supplies no word times."
        },
    )
    openai_stt_aligner_device: str = field(
        default="cpu",
        metadata={
            "help": "Device for the optional word aligner. Default: cpu; loaded only for continuous-speech boundaries."
        },
    )
