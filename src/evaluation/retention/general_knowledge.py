"""General knowledge retention evaluation.

Measures whether a domain-adapted model retains general knowledge
by testing on MMLU subsets or other general benchmarks.
"""

from __future__ import annotations

import logging

from src.evaluation.base import (
    BaseEvaluator,
    BenchmarkResult,
    EvalExample,
    EvalPrediction,
    compute_accuracy,
    score_multiple_choice,
)

logger = logging.getLogger(__name__)

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

# MMLU subsets for general knowledge retention testing
DEFAULT_MMLU_SUBSETS = [
    "abstract_algebra",
    "college_mathematics",
    "high_school_us_history",
    "logical_fallacies",
    "world_religions",
]


def load_mmlu_subset(
    subset: str, split: str = "test", max_examples: int | None = None
) -> list[EvalExample]:
    """Load a specific MMLU subset for evaluation."""
    from datasets import load_dataset

    logger.info("Loading MMLU subset: %s (%s)", subset, split)
    ds = load_dataset("cais/mmlu", subset, split=split)

    examples = []
    for i, row in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        examples.append(EvalExample(
            question=row["question"],
            choices=row["choices"],
            correct_answer=ANSWER_MAP.get(row["answer"], str(row["answer"])),
            correct_index=row["answer"],
            metadata={"source": f"mmlu_{subset}", "index": i},
        ))

    logger.info("Loaded %d examples from MMLU %s", len(examples), subset)
    return examples


def format_mcq_prompt(example: EvalExample) -> str:
    """Format a multiple-choice question as a text prompt."""
    prompt = f"Question: {example.question}\n\n"
    for i, choice in enumerate(example.choices):
        label = ANSWER_MAP.get(i, str(i))
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


class GeneralKnowledgeEvaluator(BaseEvaluator):
    """Evaluate general knowledge retention across MMLU subsets.

    Args:
        subsets: List of MMLU subset names to evaluate.
        max_examples_per_subset: Limit examples per subset.
        passing_threshold: Not used for pass/fail — retention is measured as delta.
    """

    def __init__(
        self,
        subsets: list[str] | None = None,
        max_examples_per_subset: int | None = None,
    ):
        super().__init__(name="general_knowledge", metric="accuracy", passing_threshold=0.0)
        self.subsets = subsets or DEFAULT_MMLU_SUBSETS
        self.max_examples_per_subset = max_examples_per_subset

    def evaluate(self, model, tokenizer, **kwargs) -> BenchmarkResult:
        all_predictions: list[EvalPrediction] = []
        per_subset: dict[str, float] = {}

        for subset in self.subsets:
            examples = load_mmlu_subset(subset, max_examples=self.max_examples_per_subset)
            subset_predictions: list[EvalPrediction] = []

            for example in examples:
                prompt = format_mcq_prompt(example)
                pred_idx, log_probs = score_multiple_choice(model, tokenizer, prompt, example.choices)
                pred_answer = ANSWER_MAP.get(pred_idx, str(pred_idx))
                is_correct = pred_idx == example.correct_index

                pred = EvalPrediction(
                    predicted=pred_answer,
                    correct=example.correct_answer,
                    is_correct=is_correct,
                )
                subset_predictions.append(pred)
                all_predictions.append(pred)

            subset_acc = compute_accuracy(subset_predictions)
            per_subset[subset] = round(subset_acc, 4)
            logger.info("MMLU %s: %.3f accuracy (%d examples)", subset, subset_acc, len(examples))

        overall_acc = compute_accuracy(all_predictions)
        num_correct = sum(1 for p in all_predictions if p.is_correct)

        logger.info(
            "General knowledge: %.3f overall accuracy (%d/%d)",
            overall_acc, num_correct, len(all_predictions),
        )

        return BenchmarkResult(
            name=self.name,
            metric=self.metric,
            score=overall_acc,
            num_examples=len(all_predictions),
            num_correct=num_correct,
            details={"per_subset": per_subset},
        )
