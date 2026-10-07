"""Audio-based cuts and joins for bounded transcription requests."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass

import numpy as np

from speech_to_speech.STT.word_alignment import AlignmentError, WordTiming

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AlignedAudioWindow:
    start: int
    end: int
    next_start: int
    digest: bytes
    text: str
    language: str | None
    words: tuple[WordTiming, ...] = ()


def audio_digest(audio: np.ndarray) -> bytes:
    return hashlib.sha256(np.asarray(audio).tobytes()).digest()


def _letters(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def word_spans(text: str, words: tuple[WordTiming, ...]) -> list[tuple[int, int]]:
    """Map alignment tokens back to the source text without losing punctuation.

    Require complete, ordered lexical coverage. Alignment may normalize spaces
    and punctuation; it must not skip, insert or rearrange recognized words.
    """
    chars: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text):
        normalized = _letters(character)
        chars.extend(normalized)
        positions.extend([index] * len(normalized))
    normalized_text = "".join(chars)
    offset = 0
    spans = []
    for word in words:
        token = _letters(word.text)
        if not token or not normalized_text.startswith(token, offset):
            raise AlignmentError("Word timings do not cover the transcript in order")
        spans.append((positions[offset], positions[offset + len(token) - 1] + 1))
        offset += len(token)
    if offset != len(normalized_text):
        raise AlignmentError("Word timings do not cover the complete transcript")
    return spans


def validate_words(text: str, words: tuple[WordTiming, ...], duration: float) -> None:
    previous = 0.0
    for word in words:
        if (
            not np.isfinite(word.start)
            or not np.isfinite(word.end)
            or word.start < previous
            or word.end < word.start
            or word.end > duration + 0.001
        ):
            raise AlignmentError("Word timings must be finite, ordered and within the audio")
        previous = word.end
    if words and not any(word.end > word.start for word in words):
        raise AlignmentError("Word timings contain no usable positive-duration speech")
    word_spans(text, words)


def timed_boundary(
    left_text: str,
    left_words: tuple[WordTiming, ...],
    right_text: str,
    right_words: tuple[WordTiming, ...],
    overlap_start: float,
    overlap_end: float,
) -> tuple[int, int]:
    """Join at a shared pair of words located at the same absolute audio times.

    Text agreement alone is insufficient: repeated phrases must refer to the
    same physical speech. Reject ambiguous timing matches rather than guessing.
    """
    left_spans, right_spans = word_spans(left_text, left_words), word_spans(right_text, right_words)
    anchors = []
    for i in range(len(left_words) - 1):
        old_pair = left_words[i : i + 2]
        if old_pair[0].start < overlap_start + 0.08 or old_pair[-1].end > overlap_end - 0.08:
            continue
        for j in range(len(right_words) - 1):
            pair = right_words[j : j + 2]
            if pair[0].start < overlap_start + 0.08 or pair[-1].end > overlap_end - 0.08:
                continue
            if all(
                old.end > old.start
                and new.end > new.start
                and _letters(old.text) == _letters(new.text)
                and abs(old.start - new.start) <= 0.16
                and abs(old.end - new.end) <= 0.16
                for old, new in zip(old_pair, pair)
            ):
                anchors.append((i, j))
    if not anchors or len({i - j for i, j in anchors}) != 1:
        raise AlignmentError("No unambiguous audio-timed boundary; increase overlap or use a streaming backend")
    middle = (overlap_start + overlap_end) / 2
    i, j = min(anchors, key=lambda anchor: abs(left_words[anchor[0]].start - middle))
    return left_spans[i][0], right_spans[j][0]


def merge_timed_windows(
    left_text: str,
    left_words: tuple[WordTiming, ...],
    right_text: str,
    right_words: tuple[WordTiming, ...],
    overlap_start: float,
    overlap_end: float,
) -> tuple[str, tuple[WordTiming, ...]]:
    if not left_text.strip():
        return right_text, right_words
    if not right_text.strip():
        return left_text, left_words
    left_cut, right_cut = timed_boundary(left_text, left_words, right_text, right_words, overlap_start, overlap_end)
    left_spans, right_spans = word_spans(left_text, left_words), word_spans(right_text, right_words)
    i = next(i for i, span in enumerate(left_spans) if span[0] == left_cut)
    j = next(j for j, span in enumerate(right_spans) if span[0] == right_cut)
    return left_text[:left_cut] + right_text[right_cut:], left_words[:i] + right_words[j:]


def pause_cut(audio: np.ndarray, start: int, end: int, overlap: int) -> int | None:
    """Find a near-digital-zero >=120ms gap near the end of a full window.

    Inspect fixed 10ms frames and keep the chosen cut stable across updates.
    Soft speech and noise are not proof of a pause. Only near-zero audio can
    justify a hard cut without alignment; otherwise use timed overlap.
    """
    frame = SAMPLE_RATE // 100
    search_start = max(start + frame, end - max(2 * overlap, SAMPLE_RATE))
    values = np.asarray(audio[start:end], dtype=np.float32)
    if np.issubdtype(audio.dtype, np.integer):
        values /= 32768.0
    size = len(values) // frame
    if size < 12:
        return None
    rms = np.sqrt(np.mean(values[: size * frame].reshape(-1, frame) ** 2, axis=1))
    quiet = rms <= 1e-7
    candidates = []
    run_start = None
    for index in range(size + 1):
        if index < size and quiet[index]:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            if index - run_start >= 12:
                low = max(search_start, start + run_start * frame)
                high = min(end - frame, start + index * frame)
                if low < high:
                    candidates.append((low + high) // 2)
            run_start = None
    if not candidates:
        return None
    target = end - overlap // 2
    return min(candidates, key=lambda cut: abs(cut - target))
