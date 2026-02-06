#!/usr/bin/env python3
"""Evaluate a domain-adapted model.

Runs domain benchmarks, knowledge retention tests, and terminology accuracy,
producing a JSON report comparing performance.

Usage:
    # Run legal domain benchmarks
    python scripts/evaluate_domain.py \
        --model models/legal-lora/final_adapter \
        --base-model meta-llama/Meta-Llama-3-8B-Instruct \
        --domain legal \
        --output reports/legal_eval.json

    # Run retention evaluation only
    python scripts/evaluate_domain.py \
        --model models/legal-lora/final_adapter \
        --base-model meta-llama/Meta-Llama-3-8B-Instruct \
        --eval-type retention \
        --output reports/retention_eval.json

    # Run terminology accuracy only
    python scripts/evaluate_domain.py \
        --model models/legal-lora/final_adapter \
        --base-model meta-llama/Meta-Llama-3-8B-Instruct \
        --eval-type terminology \
        --glossary data/terminology/legal_terms.json \
        --output reports/terminology_eval.json

    # Compare against saved baseline scores
    python scripts/evaluate_domain.py \
        --model models/legal-lora/final_adapter \
        --base-model meta-llama/Meta-Llama-3-8B-Instruct \
        --domain legal \
        --baseline reports/baseline_eval.json \
        --output reports/comparison.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate(args: argparse.Namespace) -> None:
    from src.config.settings import load_domain_config
    from src.evaluation.base import EvalReport
    from src.evaluation.retention.catastrophic_forgetting import (
        compare_results,
        load_baseline_scores,
    )
    from src.models.loader import load_model_for_inference

    # ── Load model ──────────────────────────────────────────────────────────
    logger.info("Loading model from: %s", args.model)
    loaded = load_model_for_inference(
        model_path=args.model,
        base_model=args.base_model,
        load_in_4bit=not args.no_quantize,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer

    # ── Load domain config ──────────────────────────────────────────────────
    domain_cfg = None
    if args.domain:
        domain_cfg = load_domain_config(args.domain)

    report = EvalReport(
        model_name=args.model,
        domain=args.domain or "unknown",
    )

    eval_types = args.eval_type or ["domain", "retention", "terminology"]

    # ── Domain benchmarks ───────────────────────────────────────────────────
    if "domain" in eval_types and args.domain:
        logger.info("Running domain benchmarks...")

        if args.domain == "legal":
            from src.evaluation.benchmarks.legal.bar_exam import BarExamEvaluator

            bar_eval = BarExamEvaluator(
                data_path=args.bar_exam_data,
                max_examples=args.max_examples,
                passing_threshold=0.7,
            )
            report.benchmarks.append(bar_eval.evaluate(model, tokenizer))

            if args.contract_data:
                from src.evaluation.benchmarks.legal.contract_analysis import (
                    ContractAnalysisEvaluator,
                )

                contract_eval = ContractAnalysisEvaluator(
                    data_path=args.contract_data,
                    max_examples=args.max_examples,
                    passing_threshold=0.8,
                )
                report.benchmarks.append(contract_eval.evaluate(model, tokenizer))

    # ── Retention evaluation ────────────────────────────────────────────────
    if "retention" in eval_types:
        logger.info("Running knowledge retention evaluation...")
        from src.evaluation.retention.general_knowledge import GeneralKnowledgeEvaluator

        retention_eval = GeneralKnowledgeEvaluator(
            max_examples_per_subset=args.max_examples,
        )
        retention_result = retention_eval.evaluate(model, tokenizer)
        report.retention.append(retention_result)

        # Compare with baseline if provided
        if args.baseline:
            logger.info("Comparing with baseline: %s", args.baseline)
            baseline_results = load_baseline_scores(args.baseline)
            max_drop = 0.05
            if domain_cfg:
                max_drop = domain_cfg.evaluation.retention.max_drop
            forgetting = compare_results(baseline_results, report.retention, max_allowed_drop=max_drop)
            report.metadata["forgetting"] = forgetting.to_dict()

    # ── Terminology evaluation ──────────────────────────────────────────────
    if "terminology" in eval_types and args.glossary:
        logger.info("Running terminology accuracy evaluation...")
        from src.evaluation.terminology.term_accuracy import TerminologyAccuracyEvaluator

        term_eval = TerminologyAccuracyEvaluator(
            glossary_path=args.glossary,
            prompts_path=args.prompts,
            top_k_terms=args.top_k_terms,
        )
        report.terminology = term_eval.evaluate(model, tokenizer)

    # ── Save report ─────────────────────────────────────────────────────────
    output_path = Path(args.output)
    report.save(output_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Evaluation Report: {args.model}")
    print("=" * 60)

    if report.benchmarks:
        print("\nDomain Benchmarks:")
        for b in report.benchmarks:
            status = "PASS" if b.passed else "FAIL"
            print(f"  {b.name}: {b.score:.3f} ({b.metric}) [{status}]")
        print(f"  Overall: {report.overall_domain_score:.3f}")

    if report.retention:
        print("\nKnowledge Retention:")
        for r in report.retention:
            print(f"  {r.name}: {r.score:.3f} ({r.metric})")

    if report.terminology:
        t = report.terminology
        status = "PASS" if t.passed else "FAIL"
        print(f"\nTerminology: {t.score:.3f} ({t.metric}) [{status}]")

    if "forgetting" in report.metadata:
        fg = report.metadata["forgetting"]
        status = "OK" if fg["all_retained"] else "WARNING"
        print(f"\nCatastrophic Forgetting: worst_drop={fg['worst_drop']:.4f} [{status}]")

    print(f"\nFull report: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a domain-adapted model")
    parser.add_argument("--model", type=str, required=True, help="Path to model or adapter")
    parser.add_argument("--base-model", type=str, help="Base model name (required for adapters)")
    parser.add_argument("--domain", type=str, help="Domain name (e.g. legal)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON report path")
    parser.add_argument(
        "--eval-type", type=str, nargs="+",
        choices=["domain", "retention", "terminology"],
        help="Which evaluations to run (default: all)",
    )
    parser.add_argument("--max-examples", type=int, help="Max examples per benchmark")
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")

    # Domain benchmark data
    parser.add_argument("--bar-exam-data", type=str, help="Path to local bar exam JSONL")
    parser.add_argument("--contract-data", type=str, help="Path to contract analysis JSONL")

    # Retention
    parser.add_argument("--baseline", type=str, help="Path to baseline eval JSON for comparison")

    # Terminology
    parser.add_argument("--glossary", type=str, help="Path to glossary JSON")
    parser.add_argument("--prompts", type=str, help="Path to evaluation prompts JSONL")
    parser.add_argument("--top-k-terms", type=int, default=200, help="Number of terms to check")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
