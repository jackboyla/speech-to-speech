from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Iterator

import numpy as np

from speech_to_speech.pipeline.console import console
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.language_detection import detect_language_from_text, warm_language_detector
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.utils.utils import resolve_device

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = (
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "es",
    "sv",
    "uk",
)


class ParakeetReduxSTTHandler(BaseSTTHandler):
    """Transcribe audio windows locally with Parakeet Redux through Photon."""

    def setup(self, device: str = "auto", gen_kwargs: dict[str, Any] | None = None) -> None:
        import moondream
        import torch

        self.device = resolve_device(device, ("cuda", "cpu"), "Parakeet Redux")
        self.model: Any = moondream.photon(
            "moondream/parakeet-redux",
            device=self.device,
            # Photon sets the process-wide PyTorch count as well as its native pool.
            cpu_threads=torch.get_num_threads(),
            single_pass_batch_capacity=1,
        )
        self._language_detector = warm_language_detector(SUPPORTED_LANGUAGES)
        self._transcribe(np.zeros(16000, dtype=np.float32))
        logger.info("Parakeet Redux ready on %s", self.device)

    def _transcribe(self, audio: np.ndarray) -> str:
        return self.model.transcribe(audio=np.asarray(audio, dtype=np.float32), sample_rate=16000)["text"].strip()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        started_at_s = perf_counter()
        text = self._transcribe(vad_audio.audio)
        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return

        # The decoder selects the language; text detection only labels its output.
        detected = detect_language_from_text(text, self._language_detector)
        console.print(f"[yellow]USER: {text}")
        self._record_final_stt(vad_audio, perf_counter() - started_at_s)
        yield Transcription(
            text=text,
            language_code=f"{detected}-auto" if detected else None,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.speech_end_at_s,
        )

    def cleanup(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            self.model = None
            model.close()
