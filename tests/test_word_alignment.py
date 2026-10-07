from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.STT.word_alignment import AlignmentCancelled, AlignmentError, QwenWordAligner, WordTiming


class _Inputs(dict):
    def to(self, *_args):
        return self


class _Processor:
    def __init__(self, timings):
        self.timings = timings
        self.requests = []

    def prepare_forced_aligner_inputs(self, **kwargs):
        self.requests.append(kwargs)
        return _Inputs(input_ids=None), [["very", "very"]]

    def decode_forced_alignment(self, **_kwargs):
        return [self.timings]


def _aligner(timings):
    aligner = QwenWordAligner()
    aligner._processor = _Processor(timings)
    aligner._model = lambda **_kwargs: SimpleNamespace(logits=None)
    aligner._model.device = "cpu"
    aligner._model.dtype = None
    aligner._model.config = SimpleNamespace(timestamp_token_id=1)
    return aligner


def _items(start=0.1, end=0.5):
    return [
        {"text": "very", "start_time": 0.0, "end_time": 0.1},
        {"text": "very", "start_time": start, "end_time": end},
    ]


@pytest.mark.parametrize("pcm", [False, True])
def test_alignment_preserves_repeated_words_and_explicit_language(pcm):
    aligner = _aligner(_items())
    audio = np.zeros(16000, dtype=np.int16 if pcm else np.float32)
    audio[:3] = [-32768, 16384, 32767] if pcm else [-1.0, 0.5, 32767 / 32768]
    assert aligner.align(audio, "very very", "EN") == [
        WordTiming("very", 0.0, 0.1),
        WordTiming("very", 0.1, 0.5),
    ]
    request = aligner._processor.requests[0]
    assert request["language"] == "English"
    assert request["sampling_rate"] == 16000
    assert request["audio"].dtype == np.float32
    np.testing.assert_array_equal(request["audio"][:3], [-1.0, 0.5, 32767 / 32768])


@pytest.mark.parametrize("start,end", [(0.09, 0.5), (0.2, 0.1), (0.1, 1.01), (float("nan"), 0.5)])
def test_alignment_rejects_uncertain_timings(start, end):
    with pytest.raises(AlignmentError):
        _aligner(_items(start, end)).align(np.zeros(16000), "very very", "en")


def test_alignment_keeps_quantized_zero_duration_words_but_rejects_all_zero():
    assert _aligner(_items(0.1, 0.1)).align(np.zeros(16000), "very very", "en") == [
        WordTiming("very", 0.0, 0.1),
        WordTiming("very", 0.1, 0.1),
    ]
    items = _items(0.0, 0.0)
    items[0]["end_time"] = 0.0
    with pytest.raises(AlignmentError, match="no usable"):
        _aligner(items).align(np.zeros(16000), "very very", "en")


def test_alignment_cancellation_and_empty_text_do_not_load():
    aligner = QwenWordAligner()
    assert aligner.align(np.zeros(16000), "", None) == []
    with pytest.raises(AlignmentCancelled):
        aligner.align(np.zeros(16000), "hello", "en", cancel_check=lambda: True)
    with pytest.raises(AlignmentError, match="known language"):
        aligner.align(np.zeros(16000), "hello", "ar")
    assert aligner._model is None
    # A cancelled caller must stop waiting without releasing another call's lock.
    checks = 0

    def cancel_during_wait():
        nonlocal checks
        checks += 1
        return checks > 1

    aligner._lock.acquire()
    try:
        with pytest.raises(AlignmentCancelled):
            aligner.align(np.zeros(16000), "hello", "en", cancel_check=cancel_during_wait)
        assert aligner._lock.locked()
        assert aligner._model is None
    finally:
        aligner._lock.release()


def test_alignment_discards_cancelled_forward_and_releases_lock():
    aligner = _aligner(_items())
    cancelled = False

    class _Model:
        device = "cpu"
        dtype = None
        config = SimpleNamespace(timestamp_token_id=1)

        def __call__(self, **_kwargs):
            nonlocal cancelled
            cancelled = True
            return SimpleNamespace(logits=None)

    aligner._model = _Model()
    with pytest.raises(AlignmentCancelled):
        aligner.align(np.zeros(16000), "very very", "en", cancel_check=lambda: cancelled)
    assert aligner._lock.acquire(blocking=False)
    aligner._lock.release()


def test_alignment_failure_does_not_expose_backend_details():
    aligner = _aligner(_items())

    def fail(**_kwargs):
        raise RuntimeError("Authorization: private-api-key; /private/cache; sensitive transcript")

    aligner._processor.prepare_forced_aligner_inputs = fail
    with pytest.raises(AlignmentError) as failure:
        aligner.align(np.zeros(16000), "very very", "en")
    assert str(failure.value) == "Word alignment failed"
    assert failure.value.__suppress_context__
