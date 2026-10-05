from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhononSTTHandlerArguments:
    """Connection settings for a separately managed Phonon speech server."""

    phonon_stt_base_url: str = field(
        default="ws://localhost:8000/v1",
        metadata={"help": "Phonon server URL including /v1, or full /v1/audio/stream URL."},
    )
    phonon_stt_api_key: Optional[str] = field(
        default=None, metadata={"help": "Optional bearer token for the Phonon endpoint."}
    )
    phonon_stt_connect_timeout: float = field(
        default=10.0, metadata={"help": "WebSocket connection timeout in seconds."}
    )
    phonon_stt_final_timeout: float = field(
        default=60.0, metadata={"help": "Maximum wait for Phonon done after the local VAD boundary."}
    )
