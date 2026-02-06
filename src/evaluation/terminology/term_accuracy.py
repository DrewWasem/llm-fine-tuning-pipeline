"""Terminology accuracy evaluation.

Tests whether a domain-adapted model correctly uses domain-specific
terminology in its generated text. Given domain prompts and a glossary
of expected terms, measures how many terms appear in the model's output.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.evaluation.base import BaseEvaluator, BenchmarkResult, generate_text

logger = logging.getLogger(__name__)


def load_glossary(path: str | Path) -> list[str]:
    """Load domain terms from a glossary JSON file.

    Supports the format produced by TerminologyExtractor:
    {"terms": [{"text": "...", ...}, ...]}
    Or a simple list: ["term1", "term2", ...]
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(t) for t in data]
    if "terms" in data:
        return [t["text"] for t in data["terms"] if isinstance(t, dict) and "text" in t]
    return []


def load_prompts(path: str | Path) -> list[dict]:
    """Load evaluation prompts from a JSONL file.

    Expected format: {"prompt": "...", "expected_terms": ["term1", ...]}
    The expected_terms field is optional — if absent, all glossary terms are checked.
    """
    path = Path(path)
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


DEFAULT_LEGAL_PROMPTS = [
    "Explain the concept of breach of fiduciary duty in corporate law.",
    "What are the key elements of a valid contract under common law?",
    "Describe the difference between actual and constructive notice in property law.",
    "Explain the doctrine of respondeat superior and its limitations.",
    "What constitutes tortious interference with a contractual relationship?",
    "Describe the requirements for establishing standing in federal court.",
    "Explain the parol evidence rule and its exceptions.",
    "What are the key considerations in a motion for summary judgment?",
    "Describe the concept of strict liability in product liability cases.",
    "Explain the differences between express and implied warranties.",
]


def count_term_matches(text: str, terms: list[str]) -> tuple[int, list[str]]:
    """Count how many glossary terms appear in the generated text.

    Returns:
        Tuple of (count of matched terms, list of matched term strings).
    """
    text_lower = text.lower()
    matched = []
    for term in terms:
        # Match whole word/phrase boundaries
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, text_lower):
            matched.append(term)
    return len(matched), matched


class TerminologyAccuracyEvaluator(BaseEvaluator):
    """Evaluate whether a model uses domain terminology correctly.

    Generates text from domain-related prompts, then checks how many
    glossary terms appear in the output.

    Args:
        glossary_path: Path to glossary JSON from TerminologyExtractor.
        prompts_path: Optional path to JSONL with evaluation prompts.
        top_k_terms: Only check the top-k most important terms.
        max_new_tokens: Max tokens per generation.
        passing_threshold: Minimum term recall to pass.
    """

    def __init__(
        self,
        glossary_path: str | Path,
        prompts_path: str | Path | None = None,
        top_k_terms: int = 200,
        max_new_tokens: int = 256,
        passing_threshold: float = 0.3,
    ):
        super().__init__(
            name="terminology_accuracy", metric="term_recall", passing_threshold=passing_threshold
        )
        self.glossary_path = glossary_path
        self.prompts_path = prompts_path
        self.top_k_terms = top_k_terms
        self.max_new_tokens = max_new_tokens

    def evaluate(self, model, tokenizer, **kwargs) -> BenchmarkResult:
        # Load glossary
        all_terms = load_glossary(self.glossary_path)
        terms = all_terms[: self.top_k_terms]
        logger.info("Evaluating with %d terms (from %d total)", len(terms), len(all_terms))

        # Load or use default prompts
        if self.prompts_path:
            prompt_records = load_prompts(self.prompts_path)
            prompts = [r["prompt"] for r in prompt_records]
        else:
            prompts = DEFAULT_LEGAL_PROMPTS

        # Generate and score
        all_matched: set[str] = set()
        per_prompt_results: list[dict] = []

        for i, prompt in enumerate(prompts):
            response = generate_text(model, tokenizer, prompt, max_new_tokens=self.max_new_tokens)
            count, matched = count_term_matches(response, terms)
            all_matched.update(matched)

            per_prompt_results.append({
                "prompt": prompt[:80],
                "terms_found": count,
                "matched_terms": matched[:10],  # truncate for report
            })

            if (i + 1) % 5 == 0:
                logger.info(
                    "Terminology progress: %d/%d prompts, %d unique terms found",
                    i + 1, len(prompts), len(all_matched),
                )

        # Compute metrics
        term_recall = len(all_matched) / len(terms) if terms else 0.0
        avg_terms_per_prompt = (
            sum(r["terms_found"] for r in per_prompt_results) / len(per_prompt_results)
            if per_prompt_results
            else 0.0
        )

        logger.info(
            "Terminology result: recall=%.3f (%d/%d terms), avg=%.1f terms/prompt",
            term_recall, len(all_matched), len(terms), avg_terms_per_prompt,
        )

        return BenchmarkResult(
            name=self.name,
            metric=self.metric,
            score=term_recall,
            num_examples=len(prompts),
            num_correct=len(all_matched),
            passing_threshold=self.passing_threshold,
            passed=term_recall >= self.passing_threshold,
            details={
                "total_terms": len(terms),
                "unique_terms_found": len(all_matched),
                "term_recall": round(term_recall, 4),
                "avg_terms_per_prompt": round(avg_terms_per_prompt, 2),
                "sample_results": per_prompt_results[:5],
            },
        )
