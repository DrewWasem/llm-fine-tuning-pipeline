#!/usr/bin/env python3
"""Generate synthetic instruction data from a domain corpus.

Produces Q&A pairs and instruction-response examples suitable for SFT training.

Usage:
    # Generate from a corpus JSONL
    python scripts/generate_synthetic.py \
        --domain legal \
        --corpus data/processed/legal_corpus/corpus.jsonl \
        --output data/processed/legal_instructions.jsonl \
        --num-samples 10000

    # Generate with terminology definitions
    python scripts/generate_synthetic.py \
        --domain legal \
        --corpus data/processed/legal_corpus/corpus.jsonl \
        --glossary data/terminology/legal_terms.json \
        --output data/processed/legal_instructions.jsonl

    # Generate Q&A pairs only
    python scripts/generate_synthetic.py \
        --domain legal \
        --corpus data/processed/legal_corpus/corpus.jsonl \
        --gen-type qa \
        --output data/processed/legal_qa.jsonl

    # Generate instruction pairs only (with quality filtering)
    python scripts/generate_synthetic.py \
        --domain legal \
        --corpus data/processed/legal_corpus/corpus.jsonl \
        --gen-type instruction \
        --min-quality 0.6 \
        --output data/processed/legal_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate(args: argparse.Namespace) -> None:
    from src.corpus.synthetic.instruction_generator import (
        InstructionGenerator,
        save_instructions,
    )
    from src.corpus.synthetic.qa_generator import QAGenerator, save_qa_pairs
    from src.corpus.synthetic.quality_filter import SyntheticQualityFilter
    from src.data.formatters.domain_chat import DOMAIN_SYSTEM_PROMPTS
    from src.data.loaders.base_loader import JSONLLoader

    # ── Load corpus ──────────────────────────────────────────────────────────
    logger.info("Loading corpus from: %s", args.corpus)
    loader = JSONLLoader(path=args.corpus, source_name=args.domain)

    system_prompt = DOMAIN_SYSTEM_PROMPTS.get(args.domain, "")
    gen_types = args.gen_type or ["qa", "instruction"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load glossary if provided ────────────────────────────────────────────
    terminology = []
    if args.glossary:
        from src.evaluation.terminology.term_accuracy import load_glossary

        terminology = load_glossary(args.glossary)
        logger.info("Loaded %d terms from glossary", len(terminology))

    # ── Quality filter ───────────────────────────────────────────────────────
    quality_filter = SyntheticQualityFilter(
        min_length=args.min_length,
        max_length=args.max_length,
        min_quality_score=args.min_quality,
    )

    total_saved = 0

    # ── Generate Q&A pairs ───────────────────────────────────────────────────
    if "qa" in gen_types:
        logger.info("Generating Q&A pairs...")
        qa_gen = QAGenerator(
            domain=args.domain,
            max_pairs_per_doc=args.max_per_doc,
            seed=args.seed,
        )

        docs = loader.load()
        max_qa = args.num_samples // len(gen_types) if args.num_samples else None
        raw_pairs = list(qa_gen.generate_from_documents(docs, max_pairs=max_qa))
        logger.info("Generated %d raw Q&A pairs", len(raw_pairs))

        filtered_pairs = quality_filter.filter_qa_pairs(raw_pairs, field="answer")
        logger.info("After filtering: %d Q&A pairs", len(filtered_pairs))

        if "instruction" in gen_types:
            # Save to separate file, will merge later
            qa_path = output_path.with_suffix(".qa.jsonl")
        else:
            qa_path = output_path

        count = save_qa_pairs(iter(filtered_pairs), qa_path, system_prompt=system_prompt, fmt=args.fmt)
        total_saved += count

    # ── Generate instruction pairs ───────────────────────────────────────────
    if "instruction" in gen_types:
        logger.info("Generating instruction-response pairs...")
        inst_gen = InstructionGenerator(
            domain=args.domain,
            max_examples_per_doc=args.max_per_doc,
            terminology=terminology,
            seed=args.seed,
        )

        docs = loader.load()
        max_inst = args.num_samples // len(gen_types) if args.num_samples else None
        raw_examples = list(inst_gen.generate_from_documents(docs, max_examples=max_inst))
        logger.info("Generated %d raw instruction examples", len(raw_examples))

        # Add terminology definitions
        if terminology:
            term_examples = inst_gen.generate_term_definitions(max_terms=args.max_terms)
            raw_examples.extend(term_examples)
            logger.info("Added %d terminology definition examples", len(term_examples))

        filtered_examples = quality_filter.filter_qa_pairs(raw_examples, field="response")
        logger.info("After filtering: %d instruction examples", len(filtered_examples))

        if "qa" in gen_types:
            inst_path = output_path.with_suffix(".inst.jsonl")
        else:
            inst_path = output_path

        count = save_instructions(
            iter(filtered_examples), inst_path, system_prompt=system_prompt, fmt=args.fmt
        )
        total_saved += count

    # ── Merge if both types were generated ───────────────────────────────────
    if "qa" in gen_types and "instruction" in gen_types:
        logger.info("Merging Q&A and instruction files...")
        qa_path = output_path.with_suffix(".qa.jsonl")
        inst_path = output_path.with_suffix(".inst.jsonl")

        with open(output_path, "w") as out:
            for part_path in [qa_path, inst_path]:
                if part_path.exists():
                    with open(part_path) as f:
                        for line in f:
                            out.write(line)
                    part_path.unlink()

        logger.info("Merged output saved to: %s", output_path)

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Synthetic Data Generation: {args.domain}")
    print("=" * 60)
    print(f"  Types: {', '.join(gen_types)}")
    print(f"  Total saved: {total_saved}")
    print(f"  Filter stats: {quality_filter.stats.to_dict()}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    stats = {
        "domain": args.domain,
        "gen_types": gen_types,
        "total_saved": total_saved,
        "filter_stats": quality_filter.stats.to_dict(),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to %s", stats_path)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic instruction data")
    parser.add_argument("--domain", type=str, required=True, help="Domain name (e.g. legal)")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument(
        "--gen-type", type=str, nargs="+",
        choices=["qa", "instruction"],
        help="Generation types (default: both)",
    )
    parser.add_argument("--num-samples", type=int, help="Target number of examples to generate")
    parser.add_argument("--max-per-doc", type=int, default=3, help="Max examples per document")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Quality filtering
    parser.add_argument("--min-length", type=int, default=50, help="Min response length (chars)")
    parser.add_argument("--max-length", type=int, default=10000, help="Max response length (chars)")
    parser.add_argument("--min-quality", type=float, default=0.5, help="Min quality score (0-1)")

    # Format
    parser.add_argument(
        "--fmt", type=str, default="chat", choices=["chat", "raw"],
        help="Output format: chat (messages list) or raw (instruction/response)",
    )

    # Terminology
    parser.add_argument("--glossary", type=str, help="Path to glossary JSON for term definitions")
    parser.add_argument("--max-terms", type=int, default=100, help="Max terms for definitions")

    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
