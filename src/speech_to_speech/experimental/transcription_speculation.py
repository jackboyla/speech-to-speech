"""Consume the existing notifier's events on a connection's asyncio loop.

Call begin from the existing turn owner. Dispatch partials to observe; dispatch
finals to finalize only after the service validates response/input ownership.
The normal service remains responsible for all protocol output and decoding.
"""

from __future__ import annotations

from speech_to_speech.experimental.turn_speculation import Action, Finalized, TurnSpeculation
from speech_to_speech.pipeline.events import (
    PartialTranscriptionEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)


class TranscriptionSpeculation:
    def __init__(self, controller: TurnSpeculation) -> None:
        self.controller = controller

    async def observe(self, event: PartialTranscriptionEvent | TranscriptionFailedEvent) -> Action:
        if self.controller.owner is None or (event.turn_id, event.turn_revision) != self.controller.owner:
            return Action.WAIT
        if isinstance(event, TranscriptionFailedEvent):
            await self.controller.cancel()
            return Action.CANCEL
        return await self.controller.update(event.delta)

    async def finalize(self, event: TranscriptionCompletedEvent) -> Finalized:
        if self.controller.owner is None or (event.turn_id, event.turn_revision) != self.controller.owner:
            return Finalized(Action.WAIT)
        return await self.controller.finalize(event.transcript)
