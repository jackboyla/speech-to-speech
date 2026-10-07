from __future__ import annotations

import numpy as np
import pytest

from speech_to_speech.STT.aligned_audio_windows import merge_timed_windows, pause_cut, validate_words
from speech_to_speech.STT.word_alignment import AlignmentError, WordTiming


def _words(start, end, text="very", shift=0):
    return tuple(WordTiming(text, i + 0.1 + shift, i + 0.4 + shift) for i in range(start, end))


def test_timed_join_preserves_one_hundred_real_repetitions():
    text, words = merge_timed_windows("very " * 60, _words(0, 60), "very " * 50, _words(50, 100), 50, 60)
    assert text.split() == ["very"] * 100
    assert words == _words(0, 100)


@pytest.mark.parametrize("shift", [0.3, -0.3])
def test_repeated_text_at_different_audio_times_cannot_be_joined(shift):
    # The two decodes disagree on which physical speech these tokens describe.
    with pytest.raises(AlignmentError):
        merge_timed_windows("very " * 60, _words(0, 60), "very " * 50, _words(50, 100, shift=shift), 50, 60)


@pytest.mark.parametrize(
    "left,right,left_words,right_words,expected",
    [
        (
            "Hello, very, very!",
            "very, very! Next?",
            ("Hello", "very", "very"),
            ("very", "very", "Next"),
            "Hello, very, very! Next?",
        ),
        ("你好世界。", "世界！再见。", ("你", "好", "世", "界"), ("世", "界", "再", "见"), "你好世界！再见。"),
    ],
)
def test_timed_join_preserves_original_punctuation_and_cjk(left, right, left_words, right_words, expected):
    offset = len(left_words) - 2
    old = tuple(WordTiming(word, i + 0.1, i + 0.4) for i, word in enumerate(left_words))
    new = tuple(WordTiming(word, i + offset + 0.1, i + offset + 0.4) for i, word in enumerate(right_words))
    text, words = merge_timed_windows(left, old, right, new, offset, len(left_words))
    assert text == expected
    validate_words(text, words, len(left_words) + len(right_words))


@pytest.mark.parametrize(
    "text,words",
    [
        ("one two", (WordTiming("one", float("nan"), 1), WordTiming("two", 1, 2))),
        ("one two", (WordTiming("one", 0, float("inf")), WordTiming("two", 1, 2))),
        ("one two", (WordTiming("one", -0.1, 1), WordTiming("two", 1, 2))),
        ("one two", (WordTiming("one", 0, 1.1), WordTiming("two", 1, 2))),
        ("one two", (WordTiming("one", 0, 1), WordTiming("two", 1, 3.1))),
        ("one two", (WordTiming("one", 0, 1),)),
        ("one two", (WordTiming("two", 0, 1), WordTiming("one", 1, 2))),
        ("one two", (WordTiming("one", 0, 1), WordTiming("extra", 1, 2))),
        ("one two", (WordTiming("one", 0.5, 0.4), WordTiming("two", 1, 2))),
    ],
)
def test_alignment_requires_finite_ordered_complete_coverage(text, words):
    with pytest.raises(AlignmentError):
        validate_words(text, words, 3)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_pause_cut_uses_real_quiet_gap_and_ignores_short_gap_or_continuous_noise(dtype):
    scale = 1 if dtype == np.float32 else 32768
    audio = np.full(16000 * 4, 0.1 * scale, dtype=dtype)
    assert pause_cut(audio, 0, len(audio), 16000) is None
    audio[48000:48800] = 0  # 50 ms does not justify a hard cut.
    assert pause_cut(audio, 0, len(audio), 16000) is None
    audio[48000:52000] = 0
    cut = pause_cut(audio, 0, len(audio), 16000)
    assert cut is not None and 48000 <= cut <= 52000


@pytest.mark.parametrize("amplitude", [1e-6, 0.001, 0.004])
def test_quiet_speech_never_provides_a_hard_pause_cut(amplitude):
    audio = np.full(16000 * 4, 0.1, dtype=np.float32)
    # Quiet speech after loud speech must not satisfy a relative RMS threshold.
    audio[48000:54000] = amplitude
    assert pause_cut(audio, 0, len(audio), 16000) is None


def test_native_zero_duration_word_is_preserved_but_cannot_anchor_join():
    old = (
        WordTiming("one", 0.1, 0.4),
        WordTiming("zero", 1, 1),
        WordTiming("two", 1.1, 1.4),
        WordTiming("three", 2.1, 2.4),
    )
    new = old[1:] + (WordTiming("four", 3.1, 3.4),)
    validate_words("one zero two three", old, 3)
    text, words = merge_timed_windows("one zero two three", old, "zero two three four", new, 1, 3)
    assert text == "one zero two three four"
    assert words == old + (WordTiming("four", 3.1, 3.4),)
    with pytest.raises(AlignmentError):
        merge_timed_windows("one zero", old[:2], "one zero", old[:2], 0, 2)


def test_all_zero_duration_alignment_cannot_establish_physical_speech():
    with pytest.raises(AlignmentError):
        validate_words("one two", (WordTiming("one", 0.1, 0.1), WordTiming("two", 0.2, 0.2)), 1)


@pytest.mark.parametrize(
    "words",
    [
        (WordTiming("one", 0.01, 0.2), WordTiming("two", 0.4, 0.6)),
        (WordTiming("one", 0.1, 0.2), WordTiming("two", 0.4, 0.99)),
    ],
)
def test_partial_words_near_overlap_edges_cannot_anchor_join(words):
    with pytest.raises(AlignmentError):
        merge_timed_windows("one two", words, "one two", words, 0, 1)


@pytest.mark.parametrize(
    "new_words",
    [
        (WordTiming("one", 0, 0.1), WordTiming("two", 0.6, 0.85)),
        (WordTiming("one", 0.1, 0.2), WordTiming("two", 0.7, 0.99)),
    ],
)
def test_new_window_partial_words_near_overlap_edges_cannot_anchor_join(new_words):
    old_words = (WordTiming("one", 0.1, 0.2), WordTiming("two", 0.6, 0.85))
    with pytest.raises(AlignmentError):
        merge_timed_windows("one two", old_words, "one two", new_words, 0, 1)
