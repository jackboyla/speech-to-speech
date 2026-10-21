from dataclasses import dataclass, field


@dataclass
class ParakeetReduxSTTHandlerArguments:
    parakeet_redux_device: str = field(
        default="auto",
        metadata={"help": "Photon device: 'auto' (CUDA, then CPU), 'cuda', or 'cpu'. Apple Silicon uses the CPU."},
    )
