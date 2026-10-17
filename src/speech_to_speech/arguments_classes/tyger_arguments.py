from dataclasses import dataclass, field


@dataclass
class TygerArguments:
    tyger_model_name: str = field(
        default="nineninesix/tyger", metadata={"help": "Tyger bundle on Hugging Face or a local directory."}
    )
    tyger_revision: str | None = field(default=None, metadata={"help": "Optional Tyger model revision."})
    tyger_device: str = field(default="cuda", metadata={"help": "Tyger inference device (cuda, cuda:N, or cpu)."})
    tyger_endpoint_threshold: float = field(default=0.91, metadata={"help": "Tyger endpoint probability threshold."})
    tyger_barge_in_threshold: float = field(default=0.5, metadata={"help": "Tyger interruption probability threshold."})
