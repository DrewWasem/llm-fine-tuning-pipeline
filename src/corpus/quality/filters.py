"""Quality filtering for domain corpus documents.

Applies configurable filters to remove low-quality documents:
- Length filters (min/max character count)
- Language detection
- Heuristic quality scoring (repetition, special char ratio, etc.)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass

from src.config.settings import QualityFilterConfig
from src.data.loaders.base_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Tracks how many documents were kept/dropped by each filter."""

    total_input: int = 0
    passed: int = 0
    dropped_too_short: int = 0
    dropped_too_long: int = 0
    dropped_low_quality: int = 0
    dropped_language: int = 0

    @property
    def drop_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return 1.0 - (self.passed / self.total_input)

    def summary(self) -> dict:
        return {
            "total_input": self.total_input,
            "passed": self.passed,
            "drop_rate": round(self.drop_rate, 4),
            "dropped_too_short": self.dropped_too_short,
            "dropped_too_long": self.dropped_too_long,
            "dropped_low_quality": self.dropped_low_quality,
            "dropped_language": self.dropped_language,
        }


def _quality_score(text: str) -> float:
    """Compute a heuristic quality score for a document (0.0 to 1.0).

    Penalizes:
    - High ratio of special/non-alphanumeric characters
    - Excessive line repetition
    - Very short average line length (table-like content)
    - High whitespace ratio
    """
    if not text:
        return 0.0

    score = 1.0

    # Penalize high special character ratio
    alnum_count = sum(1 for c in text if c.isalnum() or c.isspace())
    alnum_ratio = alnum_count / len(text)
    if alnum_ratio < 0.7:
        score -= 0.3

    # Penalize excessive line repetition
    lines = text.strip().split("\n")
    if len(lines) > 5:
        unique_lines = set(line.strip() for line in lines if line.strip())
        unique_ratio = len(unique_lines) / len(lines) if lines else 1.0
        if unique_ratio < 0.5:
            score -= 0.3

    # Penalize very short average line length (likely tables/lists)
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines:
        avg_line_len = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        if avg_line_len < 20:
            score -= 0.2

    # Penalize high whitespace ratio
    whitespace_ratio = text.count(" ") / len(text) if text else 0
    if whitespace_ratio > 0.4:
        score -= 0.1

    return max(0.0, min(1.0, score))


def _detect_language_heuristic(text: str) -> str:
    """Simple heuristic language detection based on common English patterns.

    Returns "en" if the text appears to be English, "unknown" otherwise.
    For production use, replace with a proper language detection library.
    """
    # Sample the first 1000 chars for speed
    sample = text[:1000].lower()

    # Count common English words
    english_words = {
        "the", "and", "is", "in", "to", "of", "a", "that", "it", "for",
        "was", "on", "are", "with", "as", "at", "be", "this", "have", "from",
    }
    words = re.findall(r"\b[a-z]+\b", sample)
    if not words:
        return "unknown"

    english_count = sum(1 for w in words if w in english_words)
    english_ratio = english_count / len(words)

    # If >8% of words are common English words, classify as English
    if english_ratio > 0.08:
        return "en"
    return "unknown"


class QualityFilter:
    """Applies quality filters to a stream of documents.

    Args:
        config: Quality filter configuration (min/max length, language, dedup threshold).
        min_quality_score: Minimum heuristic quality score to keep a document (0.0 to 1.0).
    """

    def __init__(self, config: QualityFilterConfig, min_quality_score: float = 0.4):
        self.config = config
        self.min_quality_score = min_quality_score
        self.stats = FilterStats()

    def _check_length(self, doc: Document) -> bool:
        length = doc.char_length
        if length < self.config.min_length:
            self.stats.dropped_too_short += 1
            return False
        if length > self.config.max_length:
            self.stats.dropped_too_long += 1
            return False
        return True

    def _check_language(self, doc: Document) -> bool:
        if not self.config.language:
            return True
        detected = _detect_language_heuristic(doc.text)
        if detected != self.config.language and detected != "unknown":
            self.stats.dropped_language += 1
            return False
        return True

    def _check_quality(self, doc: Document) -> bool:
        score = _quality_score(doc.text)
        if score < self.min_quality_score:
            self.stats.dropped_low_quality += 1
            return False
        return True

    def filter(self, documents: Iterator[Document]) -> Iterator[Document]:
        """Filter a stream of documents, yielding only those that pass all checks."""
        for doc in documents:
            self.stats.total_input += 1

            if not self._check_length(doc):
                continue
            if not self._check_language(doc):
                continue
            if not self._check_quality(doc):
                continue

            self.stats.passed += 1
            yield doc

        logger.info("Quality filter stats: %s", self.stats.summary())
