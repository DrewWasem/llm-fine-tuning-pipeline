"""Tests for general knowledge retention evaluator (no HuggingFace dataset access)."""

from src.evaluation.base import EvalExample
from src.evaluation.retention.general_knowledge import (
    ANSWER_MAP,
    DEFAULT_MMLU_SUBSETS,
    GeneralKnowledgeEvaluator,
    format_mcq_prompt,
)

# ── ANSWER_MAP tests ─────────────────────────────────────────────────────────


class TestAnswerMap:
    def test_standard_mapping(self):
        assert ANSWER_MAP[0] == "A"
        assert ANSWER_MAP[1] == "B"
        assert ANSWER_MAP[2] == "C"
        assert ANSWER_MAP[3] == "D"

    def test_four_entries(self):
        assert len(ANSWER_MAP) == 4


# ── format_mcq_prompt tests ──────────────────────────────────────────────────


class TestFormatMCQPrompt:
    def test_basic_formatting(self):
        example = EvalExample(
            question="What is 2 + 2?",
            choices=["3", "4", "5", "6"],
            correct_answer="B",
            correct_index=1,
        )
        prompt = format_mcq_prompt(example)
        assert "Question: What is 2 + 2?" in prompt
        assert "A. 3" in prompt
        assert "B. 4" in prompt
        assert "C. 5" in prompt
        assert "D. 6" in prompt
        assert prompt.strip().endswith("Answer:")

    def test_prompt_structure(self):
        example = EvalExample(
            question="Test?",
            choices=["opt1", "opt2", "opt3", "opt4"],
            correct_answer="A",
            correct_index=0,
        )
        prompt = format_mcq_prompt(example)
        lines = [line for line in prompt.strip().split("\n") if line.strip()]
        # Should have: Question line, blank, A., B., C., D., blank, Answer:
        assert lines[0].startswith("Question:")
        assert lines[-1].strip() == "Answer:"

    def test_two_choices(self):
        example = EvalExample(
            question="True or false?",
            choices=["True", "False"],
            correct_answer="A",
            correct_index=0,
        )
        prompt = format_mcq_prompt(example)
        assert "A. True" in prompt
        assert "B. False" in prompt

    def test_long_question(self):
        long_q = "This is a very long question. " * 20
        example = EvalExample(
            question=long_q,
            choices=["A", "B", "C", "D"],
            correct_answer="A",
            correct_index=0,
        )
        prompt = format_mcq_prompt(example)
        assert long_q.strip() in prompt


# ── DEFAULT_MMLU_SUBSETS tests ───────────────────────────────────────────────


class TestDefaultSubsets:
    def test_has_subsets(self):
        assert len(DEFAULT_MMLU_SUBSETS) > 0

    def test_expected_subsets_present(self):
        assert "abstract_algebra" in DEFAULT_MMLU_SUBSETS
        assert "college_mathematics" in DEFAULT_MMLU_SUBSETS

    def test_all_strings(self):
        for subset in DEFAULT_MMLU_SUBSETS:
            assert isinstance(subset, str)
            assert len(subset) > 0


# ── GeneralKnowledgeEvaluator config tests ───────────────────────────────────


class TestGeneralKnowledgeEvaluatorConfig:
    def test_default_init(self):
        evaluator = GeneralKnowledgeEvaluator()
        assert evaluator.name == "general_knowledge"
        assert evaluator.metric == "accuracy"
        assert evaluator.subsets == DEFAULT_MMLU_SUBSETS
        assert evaluator.max_examples_per_subset is None

    def test_custom_subsets(self):
        evaluator = GeneralKnowledgeEvaluator(subsets=["abstract_algebra"])
        assert evaluator.subsets == ["abstract_algebra"]

    def test_max_examples(self):
        evaluator = GeneralKnowledgeEvaluator(max_examples_per_subset=50)
        assert evaluator.max_examples_per_subset == 50

    def test_passing_threshold_zero(self):
        evaluator = GeneralKnowledgeEvaluator()
        assert evaluator.passing_threshold == 0.0
