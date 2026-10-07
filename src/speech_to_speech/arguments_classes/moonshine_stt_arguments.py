from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MoonshineSTTHandlerArguments:
    moonshine_model_name: str = field(
        default="moonshine-ai/moonshine-streaming-small",
        metadata={
            "help": (
                "The Moonshine checkpoint to load through Transformers. Streaming checkpoints come in "
                "'tiny' (fastest), 'small' and 'medium' (most accurate, slowest); the older 'moonshine-ai/moonshine-tiny' and "
                "'moonshine-ai/moonshine-base' also work, as do language checkpoints such as "
                "'moonshine-ai/moonshine-tiny-ja'. Default is 'moonshine-ai/moonshine-streaming-small'."
            )
        },
    )
    moonshine_device: str = field(
        default="auto",
        metadata={
            "help": "The device to run on. Options: 'auto' (first available of CUDA, NPU, XPU, MPS, CPU), 'cuda', 'npu', 'xpu', 'mps', 'cpu'. Default is 'auto'."
        },
    )
    moonshine_torch_dtype: str = field(
        default="auto",
        metadata={
            "help": (
                "The model dtype. 'auto' picks bfloat16 on CUDA, float16 on MPS and float32 on CPU. "
                "Also accepts 'float32', 'float16' or 'bfloat16'. Default is 'auto'."
            )
        },
    )
    moonshine_language: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The ISO code reported to the LLM and TTS. Each checkpoint knows one language, so by "
                "default this comes from the checkpoint name suffix ('-ja' gives 'ja') and is 'en' "
                "without one."
            )
        },
    )
    moonshine_max_tokens_per_second: float = field(
        default=6.5,
        metadata={
            "help": (
                "Caps the tokens generated per second of audio to stop repeat loops. "
                "Raise it for languages that use more tokens per second, such as Japanese. Default is 6.5."
            )
        },
    )
