"""Bar exam evaluation benchmark.

Evaluates legal domain knowledge using multiple-choice questions
from the MMLU professional_law subset or a local question set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

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


def load_mmlu_professional_law(split: str = "test", max_examples: int | None = None) -> list[EvalExample]:
    """Load the MMLU professional_law subset from HuggingFace.

    Returns:
        List of EvalExample with question, choices, and correct answer.
    """
    from datasets import load_dataset

    logger.info("Loading MMLU professional_law (%s split)", split)
    ds = load_dataset("cais/mmlu", "professional_law", split=split)

    examples = []
    for i, row in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        examples.append(EvalExample(
            question=row["question"],
            choices=row["choices"],
            correct_answer=ANSWER_MAP.get(row["answer"], str(row["answer"])),
            correct_index=row["answer"],
            metadata={"source": "mmlu_professional_law", "index": i},
        ))

    logger.info("Loaded %d bar exam questions", len(examples))
    return examples


def load_local_questions(path: str | Path) -> list[EvalExample]:
    """Load bar exam questions from a local JSONL file.

    Expected format per line:
    {"question": "...", "choices": ["A...", "B...", "C...", "D..."], "answer": 0}
    """
    path = Path(path)
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            obj = json.loads(line.strip())
            correct_idx = obj.get("answer", obj.get("correct_index", 0))
            examples.append(EvalExample(
                question=obj["question"],
                choices=obj.get("choices", []),
                correct_answer=ANSWER_MAP.get(correct_idx, str(correct_idx)),
                correct_index=correct_idx,
                metadata={"source": "local", "index": i},
            ))
    logger.info("Loaded %d local bar exam questions from %s", len(examples), path)
    return examples


def format_mcq_prompt(example: EvalExample) -> str:
    """Format a multiple-choice question as a text prompt."""
    prompt = f"Question: {example.question}\n\n"
    for i, choice in enumerate(example.choices):
        label = ANSWER_MAP.get(i, str(i))
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


class BarExamEvaluator(BaseEvaluator):
    """Evaluator for legal bar exam multiple-choice questions.

    Args:
        data_path: Path to local JSONL questions (if None, loads from MMLU).
        max_examples: Limit the number of evaluation examples.
        passing_threshold: Minimum accuracy to pass.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        max_examples: int | None = None,
        passing_threshold: float = 0.7,
    ):
        super().__init__(name="bar_exam", metric="accuracy", passing_threshold=passing_threshold)
        self.data_path = data_path
        self.max_examples = max_examples

    def load_examples(self) -> list[EvalExample]:
        if self.data_path:
            return load_local_questions(self.data_path)
        return load_mmlu_professional_law(max_examples=self.max_examples)

    def evaluate(self, model, tokenizer, **kwargs) -> BenchmarkResult:
        examples = self.load_examples()
        predictions: list[EvalPrediction] = []

        for i, example in enumerate(examples):
            prompt = format_mcq_prompt(example)
            pred_idx, log_probs = score_multiple_choice(model, tokenizer, prompt, example.choices)
            pred_answer = ANSWER_MAP.get(pred_idx, str(pred_idx))
            is_correct = pred_idx == example.correct_index

            predictions.append(EvalPrediction(
                predicted=pred_answer,
                correct=example.correct_answer,
                is_correct=is_correct,
                confidence=max(log_probs) if log_probs else 0.0,
            ))

            if (i + 1) % 50 == 0:
                running_acc = compute_accuracy(predictions)
                logger.info("Bar exam progress: %d/%d (acc=%.3f)", i + 1, len(examples), running_acc)

        accuracy = compute_accuracy(predictions)
        num_correct = sum(1 for p in predictions if p.is_correct)

        logger.info("Bar exam result: %.3f accuracy (%d/%d)", accuracy, num_correct, len(predictions))

        return BenchmarkResult(
            name=self.name,
            metric=self.metric,
            score=accuracy,
            num_examples=len(predictions),
            num_correct=num_correct,
            passing_threshold=self.passing_threshold,
            passed=accuracy >= self.passing_threshold,
        )
