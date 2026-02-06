"""Base evaluation framework for domain adaptation.

Provides shared abstractions for all evaluation types:
multiple-choice benchmarks, free-form generation, and metric computation.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    """A single evaluation example."""

    question: str
    choices: list[str] = field(default_factory=list)
    correct_answer: str = ""
    correct_index: int = -1
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalPrediction:
    """A model's prediction for one example."""

    predicted: str
    correct: str
    is_correct: bool
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated results from running a benchmark."""

    name: str
    metric: str
    score: float
    num_examples: int
    num_correct: int
    passing_threshold: float = 0.0
    passed: bool = False
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metric": self.metric,
            "score": round(self.score, 4),
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "passing_threshold": self.passing_threshold,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class EvalReport:
    """Full evaluation report with multiple benchmark results."""

    model_name: str
    domain: str
    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    retention: list[BenchmarkResult] = field(default_factory=list)
    terminology: BenchmarkResult | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def overall_domain_score(self) -> float:
        if not self.benchmarks:
            return 0.0
        return sum(b.score for b in self.benchmarks) / len(self.benchmarks)

    @property
    def all_benchmarks_passed(self) -> bool:
        return all(b.passed for b in self.benchmarks)

    def to_dict(self) -> dict:
        result = {
            "model_name": self.model_name,
            "domain": self.domain,
            "overall_domain_score": round(self.overall_domain_score, 4),
            "all_benchmarks_passed": self.all_benchmarks_passed,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "retention": [r.to_dict() for r in self.retention],
            "metadata": self.metadata,
        }
        if self.terminology:
            result["terminology"] = self.terminology.to_dict()
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Evaluation report saved to: %s", path)


class BaseEvaluator(ABC):
    """Abstract base for all evaluators."""

    def __init__(self, name: str, metric: str = "accuracy", passing_threshold: float = 0.7):
        self.name = name
        self.metric = metric
        self.passing_threshold = passing_threshold

    @abstractmethod
    def evaluate(self, model, tokenizer, **kwargs) -> BenchmarkResult:
        """Run the evaluation and return results."""


def compute_accuracy(predictions: list[EvalPrediction]) -> float:
    """Compute accuracy from a list of predictions."""
    if not predictions:
        return 0.0
    correct = sum(1 for p in predictions if p.is_correct)
    return correct / len(predictions)


def compute_f1(true_labels: list[str], pred_labels: list[str]) -> dict[str, float]:
    """Compute macro-averaged precision, recall, F1 from label lists."""
    if not true_labels or not pred_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    labels = sorted(set(true_labels) | set(pred_labels))
    per_label: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in labels
    }

    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            per_label[true]["tp"] += 1
        else:
            per_label[pred]["fp"] += 1
            per_label[true]["fn"] += 1

    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = per_label[label]["tp"]
        fp = per_label[label]["fp"]
        fn = per_label[label]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """Generate text from a prompt using a model.

    Args:
        model: HuggingFace causal LM.
        tokenizer: HuggingFace tokenizer.
        prompt: Input prompt string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text (excluding the prompt).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def score_multiple_choice(
    model, tokenizer, prompt: str, choices: list[str]
) -> tuple[int, list[float]]:
    """Score multiple-choice options by log-likelihood.

    Args:
        model: HuggingFace causal LM.
        tokenizer: HuggingFace tokenizer.
        prompt: The question / context.
        choices: List of answer strings.

    Returns:
        Tuple of (predicted choice index, list of log-probs per choice).
    """
    import torch

    log_probs = []
    for choice in choices:
        full_text = f"{prompt} {choice}"
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        # Negative loss = average log-likelihood
        log_probs.append(-outputs.loss.item())

    predicted_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
    return predicted_idx, log_probs
