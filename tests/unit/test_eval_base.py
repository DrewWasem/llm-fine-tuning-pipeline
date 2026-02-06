"""Tests for evaluation base framework: metrics, reports, data structures."""

import json

from src.evaluation.base import (
    BenchmarkResult,
    EvalPrediction,
    EvalReport,
    compute_accuracy,
    compute_f1,
)

# ── Metric tests ──────────────────────────────────────────────────────────────


class TestComputeAccuracy:
    def test_all_correct(self):
        preds = [EvalPrediction(predicted="A", correct="A", is_correct=True) for _ in range(10)]
        assert compute_accuracy(preds) == 1.0

    def test_all_wrong(self):
        preds = [EvalPrediction(predicted="B", correct="A", is_correct=False) for _ in range(10)]
        assert compute_accuracy(preds) == 0.0

    def test_half_correct(self):
        preds = [
            EvalPrediction(predicted="A", correct="A", is_correct=True),
            EvalPrediction(predicted="B", correct="A", is_correct=False),
        ]
        assert compute_accuracy(preds) == 0.5

    def test_empty_list(self):
        assert compute_accuracy([]) == 0.0


class TestComputeF1:
    def test_perfect_predictions(self):
        true = ["a", "b", "c", "a", "b"]
        pred = ["a", "b", "c", "a", "b"]
        result = compute_f1(true, pred)
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_all_wrong(self):
        true = ["a", "a", "a"]
        pred = ["b", "b", "b"]
        result = compute_f1(true, pred)
        assert result["f1"] == 0.0

    def test_partial_match(self):
        true = ["a", "a", "b", "b"]
        pred = ["a", "b", "b", "a"]
        result = compute_f1(true, pred)
        assert 0 < result["f1"] < 1.0

    def test_empty_lists(self):
        result = compute_f1([], [])
        assert result["f1"] == 0.0


# ── BenchmarkResult tests ────────────────────────────────────────────────────


class TestBenchmarkResult:
    def test_to_dict(self):
        result = BenchmarkResult(
            name="bar_exam",
            metric="accuracy",
            score=0.75,
            num_examples=100,
            num_correct=75,
            passing_threshold=0.7,
            passed=True,
        )
        d = result.to_dict()
        assert d["name"] == "bar_exam"
        assert d["score"] == 0.75
        assert d["passed"] is True
        assert d["num_correct"] == 75

    def test_passed_check(self):
        result = BenchmarkResult(
            name="test", metric="accuracy", score=0.5,
            num_examples=10, num_correct=5,
            passing_threshold=0.7, passed=False,
        )
        assert result.passed is False


# ── EvalReport tests ─────────────────────────────────────────────────────────


class TestEvalReport:
    def test_overall_domain_score(self):
        report = EvalReport(
            model_name="test-model",
            domain="legal",
            benchmarks=[
                BenchmarkResult(name="a", metric="accuracy", score=0.8, num_examples=10, num_correct=8),
                BenchmarkResult(name="b", metric="accuracy", score=0.6, num_examples=10, num_correct=6),
            ],
        )
        assert report.overall_domain_score == 0.7

    def test_all_benchmarks_passed(self):
        report = EvalReport(
            model_name="test",
            domain="legal",
            benchmarks=[
                BenchmarkResult(
                    name="a", metric="accuracy", score=0.8, num_examples=10,
                    num_correct=8, passing_threshold=0.7, passed=True,
                ),
                BenchmarkResult(
                    name="b", metric="accuracy", score=0.9, num_examples=10,
                    num_correct=9, passing_threshold=0.7, passed=True,
                ),
            ],
        )
        assert report.all_benchmarks_passed is True

    def test_not_all_passed(self):
        report = EvalReport(
            model_name="test",
            domain="legal",
            benchmarks=[
                BenchmarkResult(
                    name="a", metric="accuracy", score=0.8, num_examples=10,
                    num_correct=8, passing_threshold=0.7, passed=True,
                ),
                BenchmarkResult(
                    name="b", metric="accuracy", score=0.5, num_examples=10,
                    num_correct=5, passing_threshold=0.7, passed=False,
                ),
            ],
        )
        assert report.all_benchmarks_passed is False

    def test_to_dict_structure(self):
        report = EvalReport(model_name="test", domain="legal")
        d = report.to_dict()
        assert "model_name" in d
        assert "domain" in d
        assert "overall_domain_score" in d
        assert "benchmarks" in d
        assert "retention" in d

    def test_save_and_load(self, tmp_path):
        report = EvalReport(
            model_name="test",
            domain="legal",
            benchmarks=[
                BenchmarkResult(name="bar_exam", metric="accuracy", score=0.75, num_examples=100, num_correct=75),
            ],
        )
        path = tmp_path / "report.json"
        report.save(path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["model_name"] == "test"
        assert loaded["benchmarks"][0]["score"] == 0.75

    def test_empty_report(self):
        report = EvalReport(model_name="test", domain="legal")
        assert report.overall_domain_score == 0.0
        assert report.all_benchmarks_passed is True  # vacuously true
