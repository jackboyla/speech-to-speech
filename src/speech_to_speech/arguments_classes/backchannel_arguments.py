from dataclasses import dataclass, field


@dataclass
class BackchannelArguments:
    backchannel_url: str | None = field(
        default=None,
        metadata={
            "help": "Base URL of a llama.cpp server with a decision model (for example http://127.0.0.1:8093), "
            "used to tell backchannels such as 'mm-hmm' from real interruptions while the assistant is speaking. "
            "Unset (default) keeps the plain VAD barge-in."
        },
    )
    backchannel_model: str | None = field(
        default=None,
        metadata={"help": "Model name to send to the decision model server, for llama.cpp router mode."},
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
