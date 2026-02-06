"""Contract analysis evaluation benchmark.

Evaluates a model's ability to classify contract clauses by type
(e.g., termination, indemnification, limitation of liability).
Uses either a local annotated dataset or generates prompts
from contract excerpts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.evaluation.base import (
    BaseEvaluator,
    BenchmarkResult,
    compute_f1,
    generate_text,
)

logger = logging.getLogger(__name__)

# Standard contract clause categories
CLAUSE_TYPES = [
    "termination",
    "indemnification",
    "limitation_of_liability",
    "confidentiality",
    "non_compete",
    "governing_law",
    "force_majeure",
    "assignment",
    "dispute_resolution",
    "intellectual_property",
]


def load_contract_examples(path: str | Path) -> list[dict]:
    """Load contract analysis examples from JSONL.

    Expected format:
    {"text": "clause text...", "label": "termination", "context": "optional full contract context"}
    """
    path = Path(path)
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info("Loaded %d contract analysis examples from %s", len(examples), path)
    return examples


def format_classification_prompt(clause_text: str, clause_types: list[str] | None = None) -> str:
    """Format a clause classification prompt."""
    types = clause_types or CLAUSE_TYPES
    type_list = ", ".join(types)
    return (
        f"Classify the following contract clause into one of these categories: {type_list}\n\n"
        f"Clause: {clause_text}\n\n"
        f"Category:"
    )


def parse_classification_response(response: str, valid_labels: list[str] | None = None) -> str:
    """Extract a clause type label from model response."""
    response_lower = response.lower().strip()
    labels = valid_labels or CLAUSE_TYPES

    # Try exact match first
    for label in labels:
        if label in response_lower:
            return label

    # Try matching with underscores removed
    for label in labels:
        if label.replace("_", " ") in response_lower:
            return label

    # Fallback: return the first word if it looks like a label
    first_word = response_lower.split()[0] if response_lower else ""
    for label in labels:
        if first_word.startswith(label[:4]):
            return label

    return "unknown"


class ContractAnalysisEvaluator(BaseEvaluator):
    """Evaluator for contract clause classification.

    Args:
        data_path: Path to JSONL with labeled contract clauses.
        clause_types: List of valid clause type labels.
        max_examples: Limit evaluation examples.
        passing_threshold: Minimum F1 to pass.
    """

    def __init__(
        self,
        data_path: str | Path,
        clause_types: list[str] | None = None,
        max_examples: int | None = None,
        passing_threshold: float = 0.8,
    ):
        super().__init__(
            name="contract_analysis", metric="f1", passing_threshold=passing_threshold
        )
        self.data_path = data_path
        self.clause_types = clause_types or CLAUSE_TYPES
        self.max_examples = max_examples

    def evaluate(self, model, tokenizer, **kwargs) -> BenchmarkResult:
        examples = load_contract_examples(self.data_path)
        if self.max_examples:
            examples = examples[: self.max_examples]

        true_labels: list[str] = []
        pred_labels: list[str] = []

        for i, example in enumerate(examples):
            clause_text = example["text"]
            true_label = example["label"]

            prompt = format_classification_prompt(clause_text, self.clause_types)
            response = generate_text(model, tokenizer, prompt, max_new_tokens=32)
            pred_label = parse_classification_response(response, self.clause_types)

            true_labels.append(true_label)
            pred_labels.append(pred_label)

            if (i + 1) % 50 == 0:
                partial_f1 = compute_f1(true_labels, pred_labels)
                logger.info(
                    "Contract analysis progress: %d/%d (F1=%.3f)",
                    i + 1, len(examples), partial_f1["f1"],
                )

        metrics = compute_f1(true_labels, pred_labels)
        f1_score = metrics["f1"]
        num_correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)

        logger.info(
            "Contract analysis result: F1=%.3f, Precision=%.3f, Recall=%.3f",
            f1_score, metrics["precision"], metrics["recall"],
        )

        return BenchmarkResult(
            name=self.name,
            metric=self.metric,
            score=f1_score,
            num_examples=len(true_labels),
            num_correct=num_correct,
            passing_threshold=self.passing_threshold,
            passed=f1_score >= self.passing_threshold,
            details={
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(f1_score, 4),
            },
        )
