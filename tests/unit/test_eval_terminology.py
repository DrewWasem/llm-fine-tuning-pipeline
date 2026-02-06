"""Tests for terminology accuracy evaluation."""

import json
from pathlib import Path

from src.evaluation.terminology.term_accuracy import (
    count_term_matches,
    load_glossary,
    load_prompts,
)

# ── count_term_matches tests ──────────────────────────────────────────────────


class TestCountTermMatches:
    def test_basic_match(self):
        text = "The court applied the doctrine of respondeat superior in this case."
        terms = ["respondeat superior", "doctrine", "fiduciary duty"]
        count, matched = count_term_matches(text, terms)
        assert count == 2
        assert "respondeat superior" in matched
        assert "doctrine" in matched

    def test_no_matches(self):
        text = "The weather is nice today."
        terms = ["respondeat superior", "fiduciary duty"]
        count, matched = count_term_matches(text, terms)
        assert count == 0
        assert matched == []

    def test_case_insensitive(self):
        text = "BREACH OF FIDUCIARY DUTY was established."
        terms = ["breach of fiduciary duty"]
        count, matched = count_term_matches(text, terms)
        assert count == 1

    def test_word_boundary_matching(self):
        text = "The contract was breached."
        terms = ["contract", "contractual"]  # "contractual" should NOT match
        count, matched = count_term_matches(text, terms)
        assert count == 1
        assert "contract" in matched
        assert "contractual" not in matched

    def test_empty_text(self):
        count, matched = count_term_matches("", ["term1", "term2"])
        assert count == 0

    def test_empty_terms(self):
        count, matched = count_term_matches("Some text here", [])
        assert count == 0


# ── load_glossary tests ──────────────────────────────────────────────────────


class TestLoadGlossary:
    def test_load_extractor_format(self, tmp_path: Path):
        path = tmp_path / "glossary.json"
        data = {
            "num_terms": 3,
            "terms": [
                {"text": "fiduciary duty", "frequency": 100},
                {"text": "breach of contract", "frequency": 80},
                {"text": "respondeat superior", "frequency": 50},
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)

        terms = load_glossary(path)
        assert len(terms) == 3
        assert "fiduciary duty" in terms

    def test_load_simple_list(self, tmp_path: Path):
        path = tmp_path / "glossary.json"
        data = ["term1", "term2", "term3"]
        with open(path, "w") as f:
            json.dump(data, f)

        terms = load_glossary(path)
        assert terms == ["term1", "term2", "term3"]

    def test_load_empty(self, tmp_path: Path):
        path = tmp_path / "glossary.json"
        with open(path, "w") as f:
            json.dump({"terms": []}, f)

        terms = load_glossary(path)
        assert terms == []


# ── load_prompts tests ────────────────────────────────────────────────────────


class TestLoadPrompts:
    def test_load_prompts(self, tmp_path: Path):
        path = tmp_path / "prompts.jsonl"
        records = [
            {"prompt": "Explain fiduciary duty.", "expected_terms": ["fiduciary"]},
            {"prompt": "What is estoppel?"},
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        prompts = load_prompts(path)
        assert len(prompts) == 2
        assert prompts[0]["prompt"] == "Explain fiduciary duty."
        assert "expected_terms" in prompts[0]
