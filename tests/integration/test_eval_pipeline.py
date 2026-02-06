"""Integration tests for the evaluation pipeline data flow.

Tests: report building → retention comparison → terminology loading → JSON output.
No model inference — tests the data structures and wiring.
"""

import json
from pathlib import Path

from pytest import approx

from src.evaluation.base import BenchmarkResult, EvalReport
from src.evaluation.retention.catastrophic_forgetting import (
    compare_results,
    load_baseline_scores,
)
from src.evaluation.terminology.term_accuracy import count_term_matches, load_glossary


class TestEvalReportPipeline:
    """Test building a complete evaluation report with all components."""

    def test_full_report_with_benchmarks_and_retention(self, tmp_path: Path):
        """Build a report with domain benchmarks and retention results."""
        report = EvalReport(
            model_name="legal-lora-test",
            domain="legal",
            benchmarks=[
                BenchmarkResult(
                    name="bar_exam", metric="accuracy", score=0.75,
                    num_examples=100, num_correct=75,
                    passing_threshold=0.7, passed=True,
                ),
                BenchmarkResult(
                    name="contract_analysis", metric="f1", score=0.82,
                    num_examples=50, num_correct=41,
                    passing_threshold=0.8, passed=True,
                ),
            ],
            retention=[
                BenchmarkResult(
                    name="general_knowledge", metric="accuracy", score=0.68,
                    num_examples=200, num_correct=136,
                ),
            ],
        )

        assert report.overall_domain_score == approx(0.785)
        assert report.all_benchmarks_passed is True

        # Save and reload
        path = tmp_path / "eval_report.json"
        report.save(path)

        with open(path) as f:
            data = json.load(f)

        assert data["model_name"] == "legal-lora-test"
        assert data["domain"] == "legal"
        assert len(data["benchmarks"]) == 2
        assert len(data["retention"]) == 1
        assert data["overall_domain_score"] == approx(0.785, abs=0.001)

    def test_report_with_terminology(self, tmp_path: Path):
        """Build a report including terminology evaluation."""
        report = EvalReport(
            model_name="test-model",
            domain="legal",
            terminology=BenchmarkResult(
                name="terminology_accuracy", metric="term_recall", score=0.45,
                num_examples=10, num_correct=90,
                passing_threshold=0.3, passed=True,
            ),
        )

        path = tmp_path / "report.json"
        report.save(path)

        with open(path) as f:
            data = json.load(f)

        assert "terminology" in data
        assert data["terminology"]["score"] == 0.45

    def test_report_with_forgetting_metadata(self, tmp_path: Path):
        """Build a report with forgetting comparison in metadata."""
        base_results = [
            BenchmarkResult(name="mmlu", metric="accuracy", score=0.72, num_examples=100, num_correct=72),
        ]
        adapted_results = [
            BenchmarkResult(name="mmlu", metric="accuracy", score=0.69, num_examples=100, num_correct=69),
        ]

        forgetting = compare_results(base_results, adapted_results, max_allowed_drop=0.05)
        assert forgetting.all_retained is True

        report = EvalReport(
            model_name="test-model",
            domain="legal",
            retention=adapted_results,
            metadata={"forgetting": forgetting.to_dict()},
        )

        path = tmp_path / "report.json"
        report.save(path)

        with open(path) as f:
            data = json.load(f)

        assert data["metadata"]["forgetting"]["all_retained"] is True
        assert data["metadata"]["forgetting"]["worst_drop"] == approx(0.03)


class TestBaselineComparisonPipeline:
    """Test the baseline → adapted → comparison pipeline."""

    def test_save_and_load_baseline(self, tmp_path: Path):
        """Save a baseline report, reload it, and compare with adapted results."""
        # Save baseline
        baseline_data = {
            "benchmarks": [
                {"name": "mmlu", "score": 0.75, "metric": "accuracy"},
                {"name": "hellaswag", "score": 0.80, "metric": "accuracy"},
            ]
        }
        baseline_path = tmp_path / "baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline_data, f)

        # Load baseline
        base_results = load_baseline_scores(baseline_path)
        assert len(base_results) == 2

        # Adapted results (slight drop)
        adapted_results = [
            BenchmarkResult(name="mmlu", metric="accuracy", score=0.73, num_examples=100, num_correct=73),
            BenchmarkResult(name="hellaswag", metric="accuracy", score=0.78, num_examples=100, num_correct=78),
        ]

        # Compare
        forgetting = compare_results(base_results, adapted_results, max_allowed_drop=0.05)
        assert len(forgetting.results) == 2
        assert forgetting.all_retained is True
        assert forgetting.worst_drop == approx(0.02)

    def test_catastrophic_forgetting_detected(self, tmp_path: Path):
        """Detect when adapted model drops too much on general benchmarks."""
        base_results = [
            BenchmarkResult(name="mmlu", metric="accuracy", score=0.75, num_examples=100, num_correct=75),
        ]
        adapted_results = [
            BenchmarkResult(name="mmlu", metric="accuracy", score=0.55, num_examples=100, num_correct=55),
        ]

        forgetting = compare_results(base_results, adapted_results, max_allowed_drop=0.05)
        assert forgetting.all_retained is False
        assert forgetting.worst_drop == approx(0.20)

        # Save forgetting report
        report_path = tmp_path / "forgetting.json"
        forgetting.save(report_path)

        with open(report_path) as f:
            data = json.load(f)
        assert data["all_retained"] is False


class TestTerminologyPipeline:
    """Test glossary loading → term matching pipeline."""

    def test_glossary_to_term_matching(self, tmp_path: Path):
        """Load a glossary and check term matches in generated text."""
        # Create glossary
        glossary_path = tmp_path / "glossary.json"
        glossary_data = {
            "num_terms": 5,
            "terms": [
                {"text": "breach of contract", "frequency": 100},
                {"text": "fiduciary duty", "frequency": 90},
                {"text": "respondeat superior", "frequency": 80},
                {"text": "estoppel", "frequency": 70},
                {"text": "negligence", "frequency": 60},
            ],
        }
        with open(glossary_path, "w") as f:
            json.dump(glossary_data, f)

        # Load
        terms = load_glossary(glossary_path)
        assert len(terms) == 5

        # Simulate model output and check matches
        generated_text = (
            "The court found that the defendant's actions constituted a breach of contract. "
            "Furthermore, the defendant owed a fiduciary duty to the plaintiff. "
            "The doctrine of respondeat superior was also applicable in this case."
        )

        count, matched = count_term_matches(generated_text, terms)
        assert count == 3
        assert "breach of contract" in matched
        assert "fiduciary duty" in matched
        assert "respondeat superior" in matched
        assert "estoppel" not in matched
