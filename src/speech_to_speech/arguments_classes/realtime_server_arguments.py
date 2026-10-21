from dataclasses import dataclass, field


@dataclass
class BackchannelArguments:
    backchannel_url: str | None = field(
        default=None,
        metadata={
            "help": "Base URL of a System One decision server (for example http://127.0.0.1:8093), "
            "used to tell backchannels such as 'mm-hmm' from real interruptions while the assistant is speaking. "
            "Unset (default) keeps the plain VAD barge-in."
        },
    )
    backchannel_model: str | None = field(
        default=None,
        metadata={"help": "Model name sent to the decision server (required unless the server selects a default)."},
    )
    backchannel_threshold: float = field(
        default=0.5,
        metadata={"help": "Interruption probability at or above which overlapping speech cancels the response."},
    )
    backchannel_max_hold_ms: int = field(
        default=1500,
        metadata={
            "help": "Longest time a turn that starts during assistant playback is held before it is treated as "
            "an interruption."
        },
    )
    backchannel_timeout_ms: int = field(
        default=300,
        metadata={"help": "HTTP timeout for one classification. A failed request treats the turn as an interruption."},
    )


@dataclass
class RealtimeServerArguments(BackchannelArguments):
    host: str = field(
        default="127.0.0.1",
        metadata={
            "help": "Host interface for the Realtime server. Default is 127.0.0.1. Pass 0.0.0.0 explicitly "
            "to expose the unauthenticated API on the network."
        },
    )
    port: int = field(
        default=8765,
        metadata={"help": "Port for the Realtime HTTP/WebSocket server. Default is 8765."},
    )


@dataclass
class LocalRealtimeServerArguments(BackchannelArguments):
    port: int = field(
        default=8765,
        metadata={"help": "Loopback port for the local Realtime server and audio client. Default is 8765."},
    )
