"""Wake word gate for the packaged microphone client.

Detection runs on the client, as in Home Assistant voice satellites: the
microphone streams nothing to the server until the wake word is heard, then
streams until the conversation has been quiet for a while.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

WAKE_WORD_SAMPLE_RATE = 16000
# Measured on synthetic speech: 250 ms restores a clipped first word; 375 ms starts to leak the wake word.
WAKE_PREROLL_MS = 250
# The server confirms speech only after min_speech_ms of it, so the client
# cannot rely on speech_started alone before it stops sending. It also keeps
# sending while the microphone is loud, until it has been quiet this long...
_QUIET_BEFORE_SLEEP_S = 0.3
# ...unless the noise lasts this long without the server confirming speech.
_UNCONFIRMED_LOUD_S = 2.0
# A chunk is loud when it is this many times the noise floor, the 10th
# percentile of chunk levels over the last few seconds.
_LOUD_RATIO = 3.0
_NOISE_FLOOR_CHUNKS = 64
# Below this RMS (int16 units, about -56 dBFS) a chunk is never loud.
_MIN_LOUD_RMS = 50.0
_MISSING_EXTRA = 'Wake word detection needs the wakeword extra: pip install "speech-to-speech[wakeword]"'


class WakeWordDetector(Protocol):
    def detect(self, audio: bytes) -> bool: ...

    def reset(self) -> None: ...


def resolve_wake_word_model(wake_word: str) -> Path | str:
    """Return a built-in openWakeWord model name or a custom ``.tflite`` path."""

    try:
        from pyopen_wakeword import Model
    except ImportError as exc:
        raise RuntimeError(_MISSING_EXTRA) from exc
    builtin = {model.value for model in Model}
    if wake_word in builtin:
        return wake_word
    path = Path(wake_word).expanduser()
    if path.suffix == ".tflite" and path.is_file():
        return path
    raise ValueError(
        f"Unknown wake word {wake_word!r}. Choose one of {', '.join(sorted(builtin))}, "
        "or pass the path to an openWakeWord .tflite model."
    )


class OpenWakeWordDetector:
    """openWakeWord models run through pyopen-wakeword, with no TensorFlow install."""

    def __init__(self, wake_word: str, *, threshold: float, sample_rate: int) -> None:
        from pyopen_wakeword import Model, OpenWakeWord, OpenWakeWordFeatures

        model = resolve_wake_word_model(wake_word)
        if isinstance(model, Path):
            self._model = OpenWakeWord.from_model(model)
        else:
            self._model = OpenWakeWord.from_builtin(Model(model))
        self._features = OpenWakeWordFeatures.from_builtin()
        self._threshold = threshold
        self._resampler: Any = None
        if sample_rate != WAKE_WORD_SAMPLE_RATE:
            import soxr

            self._resampler = soxr.ResampleStream(sample_rate, WAKE_WORD_SAMPLE_RATE, 1, dtype="int16")

    def detect(self, audio: bytes) -> bool:
        if self._resampler is not None:
            audio = self._resampler.resample_chunk(np.frombuffer(audio, dtype=np.int16)).tobytes()
        if not audio:
            # The resampler can hold back a whole chunk, and the feature model rejects empty input.
            return False
        detected = False
        # Drain both generators so the feature buffers stay aligned with the audio.
        for features in self._features.process_streaming(audio):
            for probability in self._model.process_streaming(features):
                detected = detected or probability >= self._threshold
        return detected

    def reset(self) -> None:
        self._model.reset()
        self._features.reset()


class WakeWordGate:
    """Decide which microphone audio reaches the server.

    Asleep, every chunk goes to the detector and none is sent. On waking, the
    last ``WAKE_PREROLL_MS`` of audio goes out first: the detector fires a
    little after the wake word ends, and without it the first word of a
    request said in one breath gets clipped. Awake, chunks are sent until
    ``timeout_s`` passes with no user speech, no response in progress, and no
    audio playing, and the microphone has gone quiet.
    """

    def __init__(
        self,
        detector: WakeWordDetector,
        *,
        wake_word: str,
        timeout_s: float,
        sample_rate: int,
        playback_active: Callable[[], bool],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._detector = detector
        self._wake_word = wake_word
        self._timeout_s = timeout_s
        self._playback_active = playback_active
        self._clock = clock
        self._preroll = bytearray()
        self._preroll_bytes = int(sample_rate * WAKE_PREROLL_MS / 1000) * 2
        self._user_speaking = False
        self._response_active = False
        self._last_activity = 0.0
        self._levels: deque[float] = deque(maxlen=_NOISE_FLOOR_CHUNKS)
        self._loud_since: float | None = None
        self._last_loud = float("-inf")
        self.awake = False
        self._announce_sleep()

    def _announce_sleep(self) -> None:
        print(f"Say {self._wake_word!r} to start.", flush=True)

    def filter(self, chunk: bytes) -> bytes:
        """Return the audio to send for this microphone chunk, possibly empty."""

        now = self._clock()
        if self.awake:
            mic_busy = self._mic_busy(chunk, now)
            if self._user_speaking or self._response_active or self._playback_active():
                self._last_activity = now
            elif now - self._last_activity >= self._timeout_s and not mic_busy:
                self.awake = False
                self._levels.clear()
                self._loud_since = None
                self._announce_sleep()
                return b""
            return chunk
        self._preroll.extend(chunk)
        del self._preroll[: max(0, len(self._preroll) - self._preroll_bytes)]
        if not self._detector.detect(chunk):
            return b""
        # Clear the detector so the same utterance cannot wake it twice.
        self._detector.reset()
        self.awake = True
        self._last_activity = now
        print("Listening.", flush=True)
        preroll = bytes(self._preroll)
        self._preroll.clear()
        return preroll

    def _mic_busy(self, chunk: bytes, now: float) -> bool:
        """Whether the microphone may hold speech the server has not confirmed yet."""

        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        level = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
        self._levels.append(level)
        floor = float(np.percentile(self._levels, 10))
        if level > max(_MIN_LOUD_RMS, floor * _LOUD_RATIO):
            if self._loud_since is None:
                self._loud_since = now
            self._last_loud = now
        elif now - self._last_loud >= _QUIET_BEFORE_SLEEP_S:
            self._loud_since = None
        return self._loud_since is not None and now - self._loud_since < _UNCONFIRMED_LOUD_S

    def handle_event(self, event: Any) -> None:
        if event.type == "input_audio_buffer.speech_started":
            self._user_speaking = True
        elif event.type == "input_audio_buffer.speech_stopped":
            self._user_speaking = False
        elif event.type == "response.created":
            self._response_active = True
        elif event.type == "response.done":
            self._response_active = False
        else:
            return
        self._last_activity = self._clock()
