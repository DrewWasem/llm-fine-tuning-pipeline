"""Tests for knowledge retention and catastrophic forgetting evaluation."""

import json
from pathlib import Path

from pytest import approx

from src.evaluation.base import BenchmarkResult
from src.evaluation.retention.catastrophic_forgetting import (
    ForgettingReport,
    RetentionResult,
    compare_results,
    load_baseline_scores,
)

# ── RetentionResult tests ────────────────────────────────────────────────────


class TestRetentionResult:
    def test_no_forgetting(self):
        r = RetentionResult(
            benchmark_name="mmlu",
            base_score=0.70,
            adapted_score=0.72,
            max_allowed_drop=0.05,
        )
        assert r.delta == approx(0.02)
        assert r.drop == 0.0
        assert r.retained is True

    def test_small_drop_within_threshold(self):
        r = RetentionResult(
            benchmark_name="mmlu",
            base_score=0.70,
            adapted_score=0.67,
            max_allowed_drop=0.05,
        )
        assert r.delta == approx(-0.03)
        assert r.drop == approx(0.03)
        assert r.retained is True

    def test_catastrophic_forgetting(self):
        r = RetentionResult(
            benchmark_name="mmlu",
            base_score=0.70,
            adapted_score=0.50,
            max_allowed_drop=0.05,
        )
        assert r.delta == approx(-0.20)
        assert r.drop == approx(0.20)
        assert r.retained is False

    def test_to_dict(self):
        r = RetentionResult(
            benchmark_name="mmlu", base_score=0.70, adapted_score=0.68,
        )
        d = r.to_dict()
        assert d["benchmark"] == "mmlu"
        assert d["base_score"] == 0.70
        assert d["adapted_score"] == 0.68
        assert d["delta"] == approx(-0.02)


# ── ForgettingReport tests ───────────────────────────────────────────────────


class TestForgettingReport:
    def test_all_retained(self):
        report = ForgettingReport(results=[
            RetentionResult("a", base_score=0.8, adapted_score=0.78, max_allowed_drop=0.05),
            RetentionResult("b", base_score=0.7, adapted_score=0.69, max_allowed_drop=0.05),
        ])
        assert report.all_retained is True
        assert report.worst_drop == approx(0.02)

    def test_not_all_retained(self):
        report = ForgettingReport(results=[
            RetentionResult("a", base_score=0.8, adapted_score=0.78, max_allowed_drop=0.05),
            RetentionResult("b", base_score=0.7, adapted_score=0.50, max_allowed_drop=0.05),
        ])
        assert report.all_retained is False
        assert report.worst_drop == approx(0.20)

    def test_average_delta(self):
        report = ForgettingReport(results=[
            RetentionResult("a", base_score=0.8, adapted_score=0.78),
            RetentionResult("b", base_score=0.7, adapted_score=0.72),
        ])
        # deltas: -0.02, +0.02 → avg = 0.0
        assert report.average_delta == approx(0.0)

    def test_empty_report(self):
        report = ForgettingReport()
        assert report.all_retained is True
        assert report.worst_drop == 0.0
        assert report.average_delta == 0.0

    def test_save_and_load(self, tmp_path: Path):
        report = ForgettingReport(results=[
            RetentionResult("mmlu", base_score=0.75, adapted_score=0.73),
        ])
        path = tmp_path / "forgetting.json"
        report.save(path)

        with open(path) as f:
            data = json.load(f)
        assert data["all_retained"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["benchmark"] == "mmlu"


# ── compare_results tests ────────────────────────────────────────────────────


class TestCompareResults:
    def test_basic_comparison(self):
        base = [
            BenchmarkResult(name="mmlu", metric="accuracy", score=0.75, num_examples=100, num_correct=75),
            BenchmarkResult(name="hellaswag", metric="accuracy", score=0.80, num_examples=100, num_correct=80),
        ]
        adapted = [
            BenchmarkResult(name="mmlu", metric="accuracy", score=0.73, num_examples=100, num_correct=73),
            BenchmarkResult(name="hellaswag", metric="accuracy", score=0.78, num_examples=100, num_correct=78),
        ]
        report = compare_results(base, adapted, max_allowed_drop=0.05)
        assert len(report.results) == 2
        assert report.all_retained is True

    def test_missing_benchmark_skipped(self):
        base = [
            BenchmarkResult(name="mmlu", metric="accuracy", score=0.75, num_examples=100, num_correct=75),
            BenchmarkResult(name="hellaswag", metric="accuracy", score=0.80, num_examples=100, num_correct=80),
        ]
        adapted = [
            BenchmarkResult(name="mmlu", metric="accuracy", score=0.73, num_examples=100, num_correct=73),
        ]
        report = compare_results(base, adapted)
        assert len(report.results) == 1  # hellaswag skipped


# ── load_baseline_scores tests ────────────────────────────────────────────────


class TestLoadBaselineScores:
    def test_load_from_benchmarks_key(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        data = {
            "benchmarks": [
                {"name": "mmlu", "score": 0.75, "metric": "accuracy"},
                {"name": "hellaswag", "score": 0.80},
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f)

        results = load_baseline_scores(path)
        assert len(results) == 2
        assert results[0].name == "mmlu"
        assert results[0].score == 0.75

    def test_load_from_retention_key(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        data = {
            "retention": [
                {"name": "mmlu", "score": 0.72, "num_examples": 50},
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f)

        results = load_baseline_scores(path)
        assert len(results) == 1
        assert results[0].score == 0.72
