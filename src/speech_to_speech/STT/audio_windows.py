from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletedAudioWindow:
    digest: bytes
    text: str
    language: str | None


def reconcile_overlap(prefix: str, tail: str) -> tuple[str, bool]:
    """Replace an exactly matching boundary with the newer window's text.

    Match at most 64 words (or individual CJK characters). An exact match needs
    two tokens; a near-edge anchor needs three and tolerates two clipped tokens.
    On disagreement keep both transcripts. This is not a word alignment guarantee.
    """
    if not prefix:
        return tail, True
    if not tail:
        return prefix, True
    pattern = r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\W_]+(?:['’][^\W_]+)*"
    left = list(re.finditer(pattern, prefix))[-64:]
    right = list(re.finditer(pattern, tail))[:64]

    def key(token: re.Match[str]) -> str:
        return unicodedata.normalize("NFKC", token.group()).casefold().replace("’", "'")

    left_keys = [key(token) for token in left]
    right_keys = [key(token) for token in right]
    exact = [size for size in range(2, min(len(left), len(right)) + 1) if left_keys[-size:] == right_keys[:size]]
    # Repeated speech can fit many overlap lengths. Text alone cannot select the
    # right one, so prefer duplicated words over deleting spoken repetitions.
    if len(exact) == 1:
        return prefix[: left[-exact[0]].start()] + tail, True
    # A window can start/end inside a word, or disagree on a clipped last word.
    # Accept a longer exact anchor close to both edges. Preserve the older text
    # before it, and let the new window replace the anchor and the revisable end.
    anchors: list[tuple[int, int]] = []
    for size in range(min(len(left), len(right)), 2, -1):
        for trailing in range(3):
            left_start = len(left) - trailing - size
            if left_start < 0:
                continue
            for leading in range(3):
                if left_keys[left_start : left_start + size] == right_keys[leading : leading + size]:
                    anchors.append((left_start, leading))
    if not exact and anchors and len({old - new for old, new in anchors}) == 1:
        left_start, leading = anchors[0]
        return prefix[: left[left_start].start()] + tail[right[leading].start() :], True
    separator = "" if prefix[-1:].isspace() or tail[:1].isspace() else " "
    return prefix + separator + tail, False
