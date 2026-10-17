"""Combined Tyger transcription, endpointing, and interruption detection."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from queue import Queue
from time import perf_counter
from typing import Any

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent
from speech_to_speech.pipeline.handler_types import STTOut
from speech_to_speech.pipeline.messages import DuplexAudio, PartialTranscription, Transcription, TranscriptionFailure
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.transcript_logging import log_exception

logger = logging.getLogger(__name__)
SR = 16000
HOP = 5120
FRAME = 1280


class TygerInputHandler(BaseHandler[DuplexAudio, STTOut]):
    def setup(
        self,
        text_output_queue: Queue[Any],
        speculative_turns: SpeculativeTurnTracker,
        model_name: str = "nineninesix/tyger",
        revision: str | None = None,
        device: str = "cuda",
        endpoint_threshold: float = 0.91,
        barge_in_threshold: float = 0.5,
        enable_live_transcription: bool = False,
        live_transcription_update_interval: float = 0.8,
        gen_kwargs: dict | None = None,
    ) -> None:
        from speech_to_speech.STT.tyger_model import TygerModel

        for value in (endpoint_threshold, barge_in_threshold):
            if not 0 < value <= 1:
                raise ValueError("Tyger thresholds must lie in (0, 1].")
        self.text_output_queue = text_output_queue
        self.speculative_turns = speculative_turns
        self.endpoint_threshold = endpoint_threshold
        self.barge_in_threshold = barge_in_threshold
        self.partial_interval = max(0.8, live_transcription_update_interval) if enable_live_transcription else None
        self.model = TygerModel(model_name, revision=revision, device=device)
        self.on_session_end()

    def on_session_end(self) -> None:
        self.model.reset()
        self.speculative_turns.reset()
        self._user = bytearray()
        self._agent = bytearray()
        self._active = np.zeros(0, dtype=np.bool_)
        self._samples = 0
        self._pre = np.zeros(0, dtype=np.float32)
        self._speaking = False
        self._yielded = False
        self._clear_turn()

    def _clear_turn(self) -> None:
        self._turn: tuple[str, int] | None = None
        self._yielded = False
        self._audio = np.zeros(0, dtype=np.float32)
        self._texts: list[str] = []
        self._voiced = 0
        self._quiet = 0
        self._warm = 0
        self._endpoint_run = 0
        self._barge_run = 0
        self._turn_start = self._samples
        self._partial_at = self._samples
        self._last_voiced = self._samples

    def _start_turn(self, audio: np.ndarray, start: int) -> None:
        self._turn = self.speculative_turns.start_turn()
        self._turn_start = start
        self._audio = audio.copy()
        self.text_output_queue.put(
            SpeechStartedEvent(audio_start_ms=start * 1000 // SR, turn_id=self._turn[0], turn_revision=self._turn[1])
        )

    def _finish(self) -> Iterator[STTOut]:
        assert self._turn is not None
        turn_id, revision = self._turn
        end = self._samples
        stopped_at = perf_counter() - (end - self._last_voiced) / SR
        self.speculative_turns.segment_finalized(end * 1000 // SR)
        self.text_output_queue.put(
            SpeechStoppedEvent(
                duration_s=(end - self._turn_start) / SR,
                audio_end_ms=end * 1000 // SR,
                turn_id=turn_id,
                turn_revision=revision,
            )
        )
        try:
            text = " ".join(t for t in [*self._texts, self.model.transcribe(self._audio)] if t)
            yield Transcription(
                text=text, language_code="en", turn_id=turn_id, turn_revision=revision, speech_stopped_at_s=stopped_at
            )
        except Exception as exc:
            log_exception(logger, "Tyger transcription failed", exc)
            yield TranscriptionFailure(message="Tyger transcription failed", turn_id=turn_id, turn_revision=revision)
        finally:
            self._clear_turn()
            self._pre = np.zeros(0, dtype=np.float32)
            self.model.reset_controller()

    def process(self, item: DuplexAudio) -> Iterator[STTOut]:
        self._user.extend(item.audio)
        self._agent.extend(item.playback)
        self._active = np.concatenate(
            [self._active, np.full(len(item.audio) // 2, item.playback_active, dtype=np.bool_)]
        )
        while len(self._user) >= HOP * 2:
            user = np.frombuffer(bytes(self._user[: HOP * 2]), dtype="<i2").astype(np.float32) / 32768
            agent = np.frombuffer(bytes(self._agent[: HOP * 2]), dtype="<i2").astype(np.float32) / 32768
            active = bool(self._active[:HOP].any())
            del self._user[: HOP * 2]
            del self._agent[: HOP * 2]
            self._active = self._active[HOP:]
            self._samples += HOP
            if not active:
                self._yielded = False
            speaking = active and not self._yielded
            if speaking != self._speaking:
                self._warm = self._endpoint_run = self._barge_run = 0
                self._speaking = speaking
                if self._turn is None:
                    self._voiced = self._quiet = 0
            scores = self.model.step(user, agent, speaking)
            self._pre = np.concatenate([self._pre, user])[-2 * SR :]
            energy = 20 * np.log10(np.sqrt(np.mean(user.reshape(4, FRAME) ** 2, axis=1)) + 1e-9)
            if self._turn is not None:
                self._audio = np.concatenate([self._audio, user])
            else:
                self._voiced += int(np.sum(energy > -50)) * FRAME
                # Idle noise must not accumulate across unrelated sounds.
                if np.all(energy <= -50):
                    self._quiet += HOP
                    if self._quiet >= SR * 1.2:
                        self._voiced = 0
                else:
                    self._quiet = 0
            for frame, (score, db) in enumerate(zip(scores, energy)):
                self._warm += 1
                if db > -50:
                    self._last_voiced = self._samples - HOP + (frame + 1) * FRAME
                if speaking:
                    self._barge_run = (
                        self._barge_run + 1 if self._warm > 6 and score[5] >= self.barge_in_threshold else 0
                    )
                    if self._warm > 6 and self._barge_run >= 2:
                        if self._turn is None:
                            self._start_turn(self._pre, self._samples - len(self._pre))
                        self._yielded = True
                        self._barge_run = 0
                        break
                else:
                    if self._turn is None and self._voiced >= SR * 0.35:
                        self._start_turn(self._pre, self._samples - len(self._pre))
                    self._endpoint_run = (
                        self._endpoint_run + 1 if self._warm > 6 and score[7] >= self.endpoint_threshold else 0
                    )
                    if self._turn is not None and self._warm > 6 and self._endpoint_run >= 2:
                        yield from self._finish()
                        break
            if self._turn is None:
                continue
            # Transcribe bounded blocks while preserving the complete turn's text.
            if len(self._audio) >= 15 * SR:
                block, self._audio = self._audio[: 15 * SR], self._audio[15 * SR :]
                try:
                    self._texts.append(self.model.transcribe(block))
                except Exception as exc:
                    log_exception(logger, "Tyger segment transcription failed", exc)
                    turn_id, revision = self._turn
                    self.speculative_turns.segment_finalized(self._samples * 1000 // SR)
                    self.text_output_queue.put(
                        SpeechStoppedEvent(
                            audio_end_ms=self._samples * 1000 // SR, turn_id=turn_id, turn_revision=revision
                        )
                    )
                    yield TranscriptionFailure(
                        message="Tyger transcription failed", turn_id=turn_id, turn_revision=revision
                    )
                    self._clear_turn()
                    self.model.reset_controller()
                    continue
            if self.partial_interval is not None and (self._samples - self._partial_at) / SR >= self.partial_interval:
                self._partial_at = self._samples
                try:
                    text = " ".join(t for t in [*self._texts, self.model.transcribe(self._audio)] if t)
                except Exception as exc:
                    log_exception(logger, "Tyger partial transcription failed", exc)
                    continue
                yield PartialTranscription(text=text, turn_id=self._turn[0], turn_revision=self._turn[1])
