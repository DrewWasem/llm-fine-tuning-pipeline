"""Tests for domain benchmarks: bar exam prompt formatting, contract analysis parsing."""

import json
from pathlib import Path

from src.evaluation.base import EvalExample
from src.evaluation.benchmarks.legal.bar_exam import (
    ANSWER_MAP,
    format_mcq_prompt,
    load_local_questions,
)
from src.evaluation.benchmarks.legal.contract_analysis import (
    format_classification_prompt,
    load_contract_examples,
    parse_classification_response,
)

# ── Bar exam tests ────────────────────────────────────────────────────────────


class TestBarExamPromptFormatting:
    def test_format_mcq_prompt(self):
        example = EvalExample(
            question="What is consideration in contract law?",
            choices=[
                "A gift",
                "Something of value exchanged",
                "A verbal agreement",
                "A legal opinion",
            ],
            correct_answer="B",
            correct_index=1,
        )
        prompt = format_mcq_prompt(example)
        assert "Question:" in prompt
        assert "A." in prompt
        assert "B." in prompt
        assert "C." in prompt
        assert "D." in prompt
        assert "Answer:" in prompt

    def test_answer_map(self):
        assert ANSWER_MAP[0] == "A"
        assert ANSWER_MAP[1] == "B"
        assert ANSWER_MAP[2] == "C"
        assert ANSWER_MAP[3] == "D"


class TestBarExamLocalLoading:
    def test_load_local_questions(self, tmp_path: Path):
        path = tmp_path / "bar_exam.jsonl"
        questions = [
            {
                "question": "What is consideration?",
                "choices": ["A gift", "Something of value", "A promise", "An offer"],
                "answer": 1,
            },
            {
                "question": "What is estoppel?",
                "choices": ["A defense", "An offense", "A tort", "A contract"],
                "answer": 0,
            },
        ]
        with open(path, "w") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")

        examples = load_local_questions(path)
        assert len(examples) == 2
        assert examples[0].question == "What is consideration?"
        assert examples[0].correct_index == 1
        assert examples[0].correct_answer == "B"
        assert len(examples[0].choices) == 4


# ── Contract analysis tests ──────────────────────────────────────────────────


class TestContractAnalysisPrompt:
    def test_format_prompt(self):
        prompt = format_classification_prompt("This agreement may be terminated by either party.")
        assert "Classify" in prompt
        assert "termination" in prompt
        assert "Clause:" in prompt

    def test_format_prompt_custom_types(self):
        prompt = format_classification_prompt("Sample clause.", clause_types=["type_a", "type_b"])
        assert "type_a" in prompt
        assert "type_b" in prompt


class TestContractAnalysisParsing:
    def test_exact_match(self):
        assert parse_classification_response("termination") == "termination"
        assert parse_classification_response("indemnification") == "indemnification"

    def test_case_insensitive(self):
        assert parse_classification_response("TERMINATION") == "termination"
        assert parse_classification_response("Confidentiality") == "confidentiality"

    def test_embedded_match(self):
        assert parse_classification_response("This is a termination clause.") == "termination"

    def test_underscore_removal(self):
        result = parse_classification_response("limitation of liability")
        assert result == "limitation_of_liability"

    def test_unknown_response(self):
        result = parse_classification_response("I don't know what this is")
        assert result == "unknown"

    def test_empty_response(self):
        result = parse_classification_response("")
        assert result == "unknown"


class TestContractAnalysisLoading:
    def test_load_examples(self, tmp_path: Path):
        path = tmp_path / "contracts.jsonl"
        examples = [
            {"text": "This agreement may be terminated...", "label": "termination"},
            {"text": "Each party shall indemnify...", "label": "indemnification"},
        ]
        with open(path, "w") as f:
            for e in examples:
                f.write(json.dumps(e) + "\n")

        loaded = load_contract_examples(path)
        assert len(loaded) == 2
        assert loaded[0]["label"] == "termination"
