"""Decode microphone and rendered playback on the same sample grid."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.utils import StreamingPcm16Resampler
from speech_to_speech.pipeline.messages import DuplexAudio


class DuplexAudioDecoder:
    def __init__(self, source_rate: int) -> None:
        self.source_rate = source_rate
        self.user = StreamingPcm16Resampler(source_rate, 16000)
        self.playback = StreamingPcm16Resampler(source_rate, 16000)

    def append(self, raw: dict[str, Any], config: RuntimeConfig) -> DuplexAudio:
        try:
            user = base64.b64decode(raw["audio"], validate=True)
            playback = base64.b64decode(raw["playback_audio"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise ValueError("Tyger requires base64 audio and aligned playback_audio PCM16.") from exc
        active = raw.get("playback_active")
        if not isinstance(active, bool) or len(user) != len(playback) or len(user) % 2:
            raise ValueError("Tyger requires equal-length PCM16 audio/playback_audio and a boolean playback_active.")
        return DuplexAudio(
            audio=self.user.push(user),
            playback=self.playback.push(playback),
            playback_active=active,
            runtime_config=config,
        )
