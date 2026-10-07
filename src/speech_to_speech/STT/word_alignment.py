"""Optional Qwen word timing for bounded ASR joins.

Alignment locates the supplied text; it does not check whether that text was
spoken. The model loads only when a speech boundary needs word timings.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WordTiming:
    """Quantized word time; zero-duration words must not serve as join anchors."""

    text: str
    start: float
    end: float


class AlignmentError(RuntimeError):
    """Alignment is unavailable or returned unusable timings."""


class AlignmentCancelled(AlignmentError):
    """The caller no longer needs this alignment."""


_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}


def _check_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise AlignmentCancelled("Word alignment cancelled")


class QwenWordAligner:
    """Lazy, serialized Transformers forced alignment on an explicit device.

    Lock waits check cancellation every 50 ms. Loading and a single model forward
    cannot safely be interrupted; cancellation discards their results afterwards.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B-hf",
        device: str = "cpu",
        sample_rate: int = 16000,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("Qwen forced alignment requires 16000 Hz audio")
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self._lock = Lock()
        self._processor: Any = None
        self._model: Any = None

    def _load(self, cancel_check: Callable[[], bool] | None) -> None:
        if self._model is not None:
            return
        _check_cancelled(cancel_check)
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_name)
        _check_cancelled(cancel_check)
        # Avoid auto device selection: other workstation jobs may use the GPUs.
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        model = AutoModelForTokenClassification.from_pretrained(self.model_name, dtype=dtype)
        _check_cancelled(cancel_check)
        model = model.to(self.device).eval()
        _check_cancelled(cancel_check)
        self._processor, self._model = processor, model

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str | None = None,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[WordTiming]:
        _check_cancelled(cancel_check)
        if not text.strip():
            return []
        names = {name.lower(): name for name in _LANGUAGES.values()}
        language_key = (language or "").strip().lower()
        language_name = _LANGUAGES.get(language_key, names.get(language_key))
        if language_name is None:
            raise AlignmentError("Word alignment needs a supported, known language")
        audio = np.asarray(audio)
        if np.issubdtype(audio.dtype, np.integer):
            # Match the HTTP backend's signed 16-bit PCM interpretation.
            audio = np.clip(audio, -32768, 32767).astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)
        duration = audio.size / self.sample_rate
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all() or duration > 300:
            raise AlignmentError("Word alignment needs finite mono audio of at most 300 seconds")
        while not self._lock.acquire(timeout=0.05):
            _check_cancelled(cancel_check)
        try:
            _check_cancelled(cancel_check)
            self._load(cancel_check)
            import torch

            inputs, word_lists = self._processor.prepare_forced_aligner_inputs(
                audio=audio,
                transcript=text,
                language=language_name,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device, self._model.dtype)
            _check_cancelled(cancel_check)
            with torch.inference_mode():
                outputs = self._model(**inputs)
            _check_cancelled(cancel_check)
            batch = self._processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=self._model.config.timestamp_token_id,
            )
            if len(batch) != 1 or len(batch[0]) != len(word_lists[0]) or not batch[0]:
                raise AlignmentError("Word alignment returned an incomplete transcript")
            words = []
            previous_end = 0.0
            for item, expected in zip(batch[0], word_lists[0]):
                start, end = float(item["start_time"]), float(item["end_time"])
                if (
                    item["text"] != expected
                    or not expected.strip()
                    or not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < previous_end
                    or end < start
                    or end > duration
                ):
                    raise AlignmentError("Word alignment returned invalid or overlapping timings")
                words.append(WordTiming(expected, start, end))
                previous_end = end
            if not any(word.end > word.start for word in words):
                raise AlignmentError("Word alignment returned no usable word timings")
            _check_cancelled(cancel_check)
            return words
        except AlignmentError:
            raise
        except Exception:
            _check_cancelled(cancel_check)
            # Backend failures may contain credentials, local paths or text.
            raise AlignmentError("Word alignment failed") from None
        finally:
            self._lock.release()
