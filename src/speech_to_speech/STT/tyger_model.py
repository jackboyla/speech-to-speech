# TurnTaker architecture adapted from nineninesix-ai/tyger-s2s,
# app/plugins/ears/tyger/controller.py, under the Apache License 2.0.
# https://github.com/nineninesix-ai/tyger-s2s
# https://www.apache.org/licenses/LICENSE-2.0
"""Native inference for the bundled Nemotron encoder and Tyger turn controller."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

SAMPLE_RATE = 16000
HOP_SAMPLES = 5120
FRAME_SAMPLES = 1280
WINDOW_SAMPLES = 8 * SAMPLE_RATE
ENCODER_LAYER = 15


def _mlp_head(width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))


class _TurnTaker(nn.Module):
    """Keep the official module names and projection order for strict weight loading."""

    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.core = nn.GRU(width, width, layers, batch_first=True)
        self.out_norm = nn.LayerNorm(width)
        self.proj_user = nn.Linear(1024, width)
        self.proj_agent = nn.Linear(1024, width)
        self.proj_energy = nn.Linear(1, width)
        self.state_emb = nn.Embedding(5, width)
        self.in_norm = nn.LayerNorm(width)
        self.state_head = nn.Linear(width, 4)
        self.hazard_head = nn.Linear(width, 1)
        self.yield_head = _mlp_head(width)
        self.bc_head = _mlp_head(width)
        self.ep_head = _mlp_head(width)

    def forward(
        self,
        user: torch.Tensor,
        agent: torch.Tensor,
        energy: torch.Tensor,
        previous_state: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.in_norm(
            self.proj_user(user)
            + self.proj_agent(agent)
            + self.proj_energy(energy.unsqueeze(-1))
            + self.state_emb(previous_state)
        )
        output, state = self.core(fused, state)
        output = self.out_norm(output)
        return torch.cat(
            [
                self.state_head(output),
                self.hazard_head(output).sigmoid(),
                self.yield_head(output).sigmoid(),
                self.bc_head(output).sigmoid(),
                self.ep_head(output).sigmoid(),
            ],
            dim=-1,
        ), state


def _check_config(config: dict[str, Any]) -> None:
    expected = {
        "model_type": "tyger_nemotron_gru",
        "format_version": 1,
        "dtype": "float32",
        "state_names": ["LISTEN", "HOLD", "SPEAK", "BACKCHANNEL"],
        "output_order": ["state_logits", "hazard_logits", "yield_logits", "bc_logits", "endpoint_logits"],
        "controller.architecture.kind": "gru",
        "frontend.layer": ENCODER_LAYER,
        "frontend.hidden_size": 1024,
        "frontend.sample_rate": SAMPLE_RATE,
        "frontend.frame_period_s": FRAME_SAMPLES / SAMPLE_RATE,
        "frontend.live_window_s": WINDOW_SAMPLES / SAMPLE_RATE,
        "frontend.live_hop_s": HOP_SAMPLES / SAMPLE_RATE,
        "frontend.num_lookahead_tokens": 0,
        "frontend.energy_norm": "clip((dB + 50) / 25, -3, 3)",
        "streaming.live_previous_state": {"listen": 0, "speaking": 2},
    }
    for path, value in expected.items():
        actual: Any = config
        for part in path.split("."):
            actual = actual.get(part) if isinstance(actual, dict) else None
        if actual != value:
            raise ValueError(f"Unsupported Tyger bundle {path}: expected {value!r}, got {actual!r}")


def _check_digest(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Tyger bundle checksum mismatch: {path.name}")


class TygerModel:
    """Single-conversation, float32 runtime; all model files come from one bundle."""

    def __init__(
        self,
        model_name: str = "nineninesix/tyger",
        revision: str | None = None,
        device: str = "cuda",
    ) -> None:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoProcessor, GenerationConfig, Nemotron3_5AsrForRNNT

        folder = Path(model_name)
        if not folder.is_dir():
            folder = Path(
                snapshot_download(
                    model_name,
                    revision=revision,
                    allow_patterns=["config.json", "model.safetensors", "nemotron/*.json"],
                )
            )
        self.config = json.loads((folder / "config.json").read_text())
        _check_config(self.config)
        for name, digest in self.config["frontend_file_sha256"].items():
            _check_digest(folder / "nemotron" / name, digest)
        _check_digest(folder / "model.safetensors", self.config["weights_sha256"])

        self.device = torch.device(device)
        frontend_folder = folder / "nemotron"
        frontend_config = AutoConfig.from_pretrained(frontend_folder, local_files_only=True)
        encoder_config = frontend_config.encoder_config
        if (
            frontend_config.model_type != "nemotron3_5_asr"
            or encoder_config.hidden_size != 1024
            or encoder_config.num_hidden_layers < ENCODER_LAYER
            or encoder_config.subsampling_factor != 8
            or 0 not in encoder_config.supported_num_lookahead_tokens
        ):
            raise ValueError("Tyger requires the bundled Nemotron 3.5 ASR frontend on its 80 ms frame grid")
        self.processor = AutoProcessor.from_pretrained(frontend_folder, local_files_only=True)
        extractor = self.processor.feature_extractor
        if extractor.sampling_rate != SAMPLE_RATE or extractor.hop_length != 160:
            raise ValueError("Tyger requires 16 kHz audio with 10 ms mel frames")

        architecture = self.config["controller"]["architecture"]
        with torch.device("meta"):
            self.asr = Nemotron3_5AsrForRNNT(frontend_config)
            self.controller = _TurnTaker(architecture["width"], architecture["layers"])
        weights = load_file(str(folder / "model.safetensors"), device="cpu")
        if len(weights) != self.config["tensor_count"] or any(t.dtype != torch.float32 for t in weights.values()):
            raise ValueError("Tyger bundle tensor count or float32 weight layout does not match its config")
        parts: dict[str, dict[str, torch.Tensor]] = {"nemotron": {}, "turn_taker": {}}
        for key, tensor in weights.items():
            prefix, separator, name = key.partition(".")
            if not separator or prefix not in parts:
                raise ValueError(f"Unknown Tyger weight component: {key}")
            parts[prefix][name] = tensor
        self.asr.load_state_dict(parts["nemotron"], strict=True, assign=True)
        self.controller.load_state_dict(parts["turn_taker"], strict=True, assign=True)
        # Meta construction also places this nonpersistent, unbundled buffer on meta.
        positions = self.asr.encoder.encode_positions
        positions.inv_freq = positions.compute_default_relative_positional_parameters(positions.config)
        nn.Module.to(self.asr, self.device)
        self.asr.eval()
        self.controller.to(self.device).eval()
        self.asr.generation_config = GenerationConfig.from_pretrained(frontend_folder, local_files_only=True)
        self.reset()

    def reset_controller(self) -> None:
        """Forget turn decisions without removing aligned audio context."""
        self._state: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear all conversation state when a client disconnects."""
        self._windows: np.ndarray = np.empty((2, 0), dtype=np.float32)
        self._speaking: bool | None = None
        self.reset_controller()

    @torch.inference_mode()
    def step(self, user_hop: np.ndarray, agent_hop: np.ndarray, speaking: bool) -> np.ndarray:
        """Return four 80 ms decisions: state logits, then four probabilities."""
        if user_hop.shape != (HOP_SAMPLES,) or agent_hop.shape != (HOP_SAMPLES,):
            raise ValueError("Tyger needs aligned 5120-sample microphone and playback hops")
        if speaking != self._speaking:
            self.reset_controller()
            self._speaking = speaking
        self._windows = np.concatenate([self._windows, np.stack([user_hop, agent_hop])], axis=1)[:, -WINDOW_SAMPLES:]
        inputs = self.processor(
            audio=[self._windows[0], self._windows[1]],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).to(self.device)
        encoded = self.asr.encoder(
            input_features=inputs.input_features,
            attention_mask=inputs.attention_mask,
            num_lookahead_tokens=0,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[ENCODER_LAYER][:, -4:]
        if encoded.shape != (2, 4, 1024):
            raise ValueError(f"Tyger frontend returned an incompatible hop shape: {tuple(encoded.shape)}")
        frames = user_hop.reshape(4, FRAME_SAMPLES)
        energy_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1) + 1e-12) + 1e-9)
        energy = torch.as_tensor(energy_db, dtype=torch.float32, device=self.device)[None]
        previous = torch.full((1, 4), 2 if speaking else 0, dtype=torch.long, device=self.device)
        outputs, self._state = self.controller(
            encoded[0:1].float(),
            encoded[1:2].float(),
            ((energy + 50) / 25).clamp(-3, 3),
            previous,
            self._state,
        )
        return outputs[0].cpu().numpy()

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe one bounded user-audio segment with the bundled RNN-T."""
        if audio.ndim != 1:
            raise ValueError("Tyger transcription requires mono audio")
        if not audio.size:
            return ""
        inputs = self.processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        sequences = self.asr.generate(**inputs).sequences
        text = self.processor.decode(sequences[0], skip_special_tokens=True).strip()
        return re.sub(r"^<\|[^|]+\|>\s*", "", text).strip()
