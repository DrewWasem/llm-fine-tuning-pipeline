"""Tests for CLI scripts: argument parsing, help text, and validation."""

import subprocess
import sys


def _run_script(script: str, args: list[str], expect_error: bool = False) -> subprocess.CompletedProcess:
    """Run a script as a subprocess and return the result."""
    result = subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if not expect_error:
        assert result.returncode == 0, f"Script failed: {result.stderr}"
    return result


# ── build_corpus.py ──────────────────────────────────────────────────────────


class TestBuildCorpusCLI:
    def test_help(self):
        result = _run_script("scripts/build_corpus.py", ["--help"])
        assert "Build a domain corpus" in result.stdout
        assert "--domain" in result.stdout
        assert "--output" in result.stdout

    def test_missing_required_args(self):
        result = _run_script("scripts/build_corpus.py", [], expect_error=True)
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_help_shows_all_options(self):
        result = _run_script("scripts/build_corpus.py", ["--help"])
        for arg in ["--source", "--local-path", "--max-docs", "--skip-dedup", "--format"]:
            assert arg in result.stdout, f"Missing {arg} in help"


# ── extract_terminology.py ───────────────────────────────────────────────────


class TestExtractTerminologyCLI:
    def test_help(self):
        result = _run_script("scripts/extract_terminology.py", ["--help"])
        assert "Extract domain terminology" in result.stdout
        assert "--corpus" in result.stdout
        assert "--output" in result.stdout

    def test_missing_required_args(self):
        result = _run_script("scripts/extract_terminology.py", [], expect_error=True)
        assert result.returncode != 0

    def test_help_shows_all_options(self):
        result = _run_script("scripts/extract_terminology.py", ["--help"])
        for arg in ["--domain", "--min-frequency", "--top-k", "--no-spacy", "--max-docs"]:
            assert arg in result.stdout, f"Missing {arg} in help"


# ── train_domain_model.py ────────────────────────────────────────────────────


class TestTrainDomainModelCLI:
    def test_help(self):
        result = _run_script("scripts/train_domain_model.py", ["--help"])
        assert "--domain" in result.stdout
        assert "--data" in result.stdout
        assert "--output" in result.stdout

    def test_missing_required_args(self):
        result = _run_script("scripts/train_domain_model.py", [], expect_error=True)
        assert result.returncode != 0

    def test_help_shows_all_options(self):
        result = _run_script("scripts/train_domain_model.py", ["--help"])
        for arg in ["--config", "--base-model", "--merge", "--resume-from", "--general-data"]:
            assert arg in result.stdout, f"Missing {arg} in help"


# ── evaluate_domain.py ───────────────────────────────────────────────────────


class TestEvaluateDomainCLI:
    def test_help(self):
        result = _run_script("scripts/evaluate_domain.py", ["--help"])
        assert "--model" in result.stdout
        assert "--output" in result.stdout

    def test_missing_required_args(self):
        result = _run_script("scripts/evaluate_domain.py", [], expect_error=True)
        assert result.returncode != 0

    def test_help_shows_all_options(self):
        result = _run_script("scripts/evaluate_domain.py", ["--help"])
        for arg in [
            "--domain", "--eval-type", "--base-model",
            "--baseline", "--glossary", "--bar-exam-data",
        ]:
            assert arg in result.stdout, f"Missing {arg} in help"

    def test_eval_type_choices(self):
        result = _run_script("scripts/evaluate_domain.py", ["--help"])
        assert "domain" in result.stdout
        assert "retention" in result.stdout
        assert "terminology" in result.stdout


# ── generate_synthetic.py ────────────────────────────────────────────────────


class TestGenerateSyntheticCLI:
    def test_help(self):
        result = _run_script("scripts/generate_synthetic.py", ["--help"])
        assert "--domain" in result.stdout
        assert "--corpus" in result.stdout
        assert "--output" in result.stdout

    def test_missing_required_args(self):
        result = _run_script("scripts/generate_synthetic.py", [], expect_error=True)
        assert result.returncode != 0

    def test_help_shows_all_options(self):
        result = _run_script("scripts/generate_synthetic.py", ["--help"])
        for arg in [
            "--gen-type", "--num-samples", "--max-per-doc",
            "--min-quality", "--glossary", "--fmt",
        ]:
            assert arg in result.stdout, f"Missing {arg} in help"

    def test_gen_type_choices(self):
        result = _run_script("scripts/generate_synthetic.py", ["--help"])
        assert "qa" in result.stdout
        assert "instruction" in result.stdout
