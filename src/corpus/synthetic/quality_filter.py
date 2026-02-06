"""Quality filtering for synthetic generated data.

Rejects low-quality generations based on:
- Length constraints (too short or too long)
- Repetition detection (repeated phrases/sentences)
- Basic coherence checks (alphanumeric ratio, sentence structure)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Statistics from a filtering run."""

    total: int = 0
    passed: int = 0
    rejected_too_short: int = 0
    rejected_too_long: int = 0
    rejected_repetition: int = 0
    rejected_low_quality: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "rejected_too_short": self.rejected_too_short,
            "rejected_too_long": self.rejected_too_long,
            "rejected_repetition": self.rejected_repetition,
            "rejected_low_quality": self.rejected_low_quality,
        }


def _repetition_ratio(text: str, ngram_size: int = 3) -> float:
    """Compute the fraction of repeated n-gram sequences in text.

    Returns a ratio from 0.0 (no repetition) to 1.0 (all repeated).
    """
    words = text.lower().split()
    if len(words) < ngram_size * 2:
        return 0.0

    ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / total if total > 0 else 0.0


def _sentence_repetition_ratio(text: str) -> float:
    """Compute the fraction of repeated sentences."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    counts = Counter(s.lower() for s in sentences)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(sentences) if sentences else 0.0


def _quality_score(text: str) -> float:
    """Compute a basic quality score for generated text.

    Returns a score from 0.0 (low quality) to 1.0 (high quality).
    """
    if not text.strip():
        return 0.0

    score = 1.0

    # Penalize very low alpha ratio (too many special chars / numbers)
    alpha_chars = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_chars / max(len(text), 1)
    if alpha_ratio < 0.5:
        score -= 0.3

    # Penalize excessive whitespace
    whitespace_ratio = sum(1 for c in text if c.isspace()) / max(len(text), 1)
    if whitespace_ratio > 0.4:
        score -= 0.2

    # Penalize if too few complete sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2 and len(text) > 200:
        score -= 0.2

    # Penalize high n-gram repetition
    rep_ratio = _repetition_ratio(text)
    if rep_ratio > 0.3:
        score -= 0.3
    elif rep_ratio > 0.15:
        score -= 0.15

    # Penalize sentence-level repetition
    sent_rep = _sentence_repetition_ratio(text)
    if sent_rep > 0.3:
        score -= 0.3

    return max(0.0, score)


class SyntheticQualityFilter:
    """Filter synthetic generated examples for quality.

    Args:
        min_length: Minimum character length for responses.
        max_length: Maximum character length for responses.
        min_quality_score: Minimum quality score threshold (0.0–1.0).
        max_repetition_ratio: Maximum allowed n-gram repetition ratio.
    """

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 10000,
        min_quality_score: float = 0.5,
        max_repetition_ratio: float = 0.4,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_quality_score = min_quality_score
        self.max_repetition_ratio = max_repetition_ratio
        self.stats = FilterStats()

    def reset_stats(self) -> None:
        self.stats = FilterStats()

    def check(self, text: str) -> tuple[bool, str]:
        """Check if a text passes quality filters.

        Returns:
            Tuple of (passed: bool, reason: str).
        """
        self.stats.total += 1

        # Length checks
        if len(text) < self.min_length:
            self.stats.rejected_too_short += 1
            return False, "too_short"

        if len(text) > self.max_length:
            self.stats.rejected_too_long += 1
            return False, "too_long"

        # Repetition check
        rep_ratio = _repetition_ratio(text)
        if rep_ratio > self.max_repetition_ratio:
            self.stats.rejected_repetition += 1
            return False, "too_repetitive"

        # Quality score check
        score = _quality_score(text)
        if score < self.min_quality_score:
            self.stats.rejected_low_quality += 1
            return False, "low_quality"

        self.stats.passed += 1
        return True, "passed"

    def filter_qa_pairs(self, pairs, field: str = "answer"):
        """Filter a list of QAPair or InstructionExample objects.

        Args:
            pairs: List of objects with the specified text field.
            field: Attribute name to check ("answer" for QAPair, "response" for InstructionExample).

        Returns:
            List of pairs that passed filtering, with quality_score updated.
        """
        passed = []
        for pair in pairs:
            text = getattr(pair, field, "")
            ok, reason = self.check(text)
            if ok:
                pair.quality_score = _quality_score(text)
                passed.append(pair)
            else:
                logger.debug(
                    "Rejected %s (doc=%s): %s",
                    type(pair).__name__,
                    getattr(pair, "source_doc_id", "?"),
                    reason,
                )
        return passed
