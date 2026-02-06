"""Catastrophic forgetting detection.

Compares evaluation scores between a base model and its domain-adapted
variant to measure how much general knowledge was lost during adaptation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.base import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class RetentionResult:
    """Result of comparing base vs adapted model on general benchmarks."""

    benchmark_name: str
    base_score: float
    adapted_score: float
    max_allowed_drop: float = 0.05

    @property
    def delta(self) -> float:
        """Score change (negative = forgetting)."""
        return self.adapted_score - self.base_score

    @property
    def drop(self) -> float:
        """Absolute drop (positive = forgetting)."""
        return max(0.0, self.base_score - self.adapted_score)

    @property
    def retained(self) -> bool:
        """Whether the model stayed within acceptable retention bounds."""
        return self.drop <= self.max_allowed_drop

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark_name,
            "base_score": round(self.base_score, 4),
            "adapted_score": round(self.adapted_score, 4),
            "delta": round(self.delta, 4),
            "drop": round(self.drop, 4),
            "max_allowed_drop": self.max_allowed_drop,
            "retained": self.retained,
        }


@dataclass
class ForgettingReport:
    """Full catastrophic forgetting report."""

    results: list[RetentionResult] = field(default_factory=list)

    @property
    def all_retained(self) -> bool:
        return all(r.retained for r in self.results)

    @property
    def worst_drop(self) -> float:
        if not self.results:
            return 0.0
        return max(r.drop for r in self.results)

    @property
    def average_delta(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.delta for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "all_retained": self.all_retained,
            "worst_drop": round(self.worst_drop, 4),
            "average_delta": round(self.average_delta, 4),
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Forgetting report saved to: %s", path)


def compare_results(
    base_results: list[BenchmarkResult],
    adapted_results: list[BenchmarkResult],
    max_allowed_drop: float = 0.05,
) -> ForgettingReport:
    """Compare base and adapted model benchmark results.

    Args:
        base_results: Benchmark results from the base model.
        adapted_results: Benchmark results from the adapted model.
        max_allowed_drop: Maximum acceptable accuracy drop.

    Returns:
        ForgettingReport with per-benchmark retention analysis.
    """
    base_by_name = {r.name: r for r in base_results}
    adapted_by_name = {r.name: r for r in adapted_results}

    report = ForgettingReport()

    for name in base_by_name:
        if name not in adapted_by_name:
            logger.warning("Benchmark %s not found in adapted results, skipping", name)
            continue

        report.results.append(RetentionResult(
            benchmark_name=name,
            base_score=base_by_name[name].score,
            adapted_score=adapted_by_name[name].score,
            max_allowed_drop=max_allowed_drop,
        ))

    logger.info(
        "Forgetting analysis: %d benchmarks, all_retained=%s, worst_drop=%.4f",
        len(report.results),
        report.all_retained,
        report.worst_drop,
    )
    return report


def load_baseline_scores(path: str | Path) -> list[BenchmarkResult]:
    """Load previously-saved baseline scores from a JSON file.

    Expected format: {"benchmarks": [{"name": ..., "score": ..., ...}, ...]}
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    results = []
    benchmarks = data.get("benchmarks", data.get("retention", []))
    for b in benchmarks:
        results.append(BenchmarkResult(
            name=b["name"],
            metric=b.get("metric", "accuracy"),
            score=b["score"],
            num_examples=b.get("num_examples", 0),
            num_correct=b.get("num_correct", 0),
        ))
    return results
