#!/usr/bin/env python3
"""Build a domain corpus: load → filter → deduplicate → format → save.

Usage:
    python scripts/build_corpus.py --domain legal --output data/processed/legal_corpus
    python scripts/build_corpus.py --domain legal --source pile-of-law --max-docs 10000
    python scripts/build_corpus.py --domain legal --local-path data/raw/contracts.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import load_domain_config
from src.corpus.quality.deduplication import Deduplicator
from src.corpus.quality.filters import QualityFilter
from src.data.formatters.domain_chat import DomainChatFormatter
from src.data.loaders.base_loader import Document, JSONLLoader
from src.data.loaders.legal_loader import HuggingFaceLoader, LegalLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_corpus(args: argparse.Namespace) -> None:
    # Load domain config
    domain_cfg = load_domain_config(args.domain)
    logger.info("Building corpus for domain: %s", domain_cfg.name.value)

    # ── Step 1: Load documents ──────────────────────────────────────────────
    if args.local_path:
        logger.info("Loading from local file: %s", args.local_path)
        loader = JSONLLoader(path=args.local_path, source_name="local")
        documents = loader.load()
    elif args.source:
        logger.info("Loading from HuggingFace: %s", args.source)
        loader = HuggingFaceLoader(
            dataset_name=args.source,
            subset=args.subset,
            split=args.split,
            text_field=args.text_field,
            max_documents=args.max_docs,
            streaming=True,
        )
        documents = loader.load()
    else:
        # Use sources from domain config
        sources = [s.model_dump() for s in domain_cfg.corpus.sources]
        for s in sources:
            s["type"] = "huggingface"
            if args.max_docs:
                s["max_documents"] = args.max_docs
        loader = LegalLoader(sources=sources, max_documents=args.max_docs)
        documents = loader.load()

    # ── Step 2: Quality filtering ───────────────────────────────────────────
    quality_filter = QualityFilter(
        config=domain_cfg.corpus.quality_filters,
        min_quality_score=args.min_quality,
    )
    documents = quality_filter.filter(documents)

    # ── Step 3: Deduplication ───────────────────────────────────────────────
    if not args.skip_dedup:
        deduplicator = Deduplicator(
            threshold=domain_cfg.corpus.quality_filters.dedup_threshold,
        )
        documents = deduplicator.deduplicate(documents)

    # ── Step 4: Save raw corpus ─────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.jsonl"
    doc_count = 0
    total_chars = 0

    with open(corpus_path, "w") as f:
        for doc in documents:
            record = {
                "text": doc.text,
                "doc_id": doc.doc_id,
                "source": doc.source,
                "metadata": doc.metadata,
            }
            f.write(json.dumps(record) + "\n")
            doc_count += 1
            total_chars += doc.char_length

            if doc_count % 1000 == 0:
                logger.info("Processed %d documents...", doc_count)

    logger.info(
        "Saved %d documents (%d chars) to %s",
        doc_count, total_chars, corpus_path,
    )

    # ── Step 5: Format for training (optional) ──────────────────────────────
    if args.format:
        logger.info("Formatting corpus as chat examples (mode=%s)", args.format)
        formatter = DomainChatFormatter(
            domain=args.domain,
            mode=args.format,
            max_length=args.max_length,
        )

        # Re-read the saved corpus
        raw_loader = JSONLLoader(path=corpus_path, source_name=args.domain)
        formatted_path = output_dir / "formatted.jsonl"
        examples = formatter.format(raw_loader.load())
        num_examples = formatter.save_jsonl(examples, formatted_path)
        logger.info("Saved %d formatted examples to %s", num_examples, formatted_path)

    # ── Step 6: Save stats ──────────────────────────────────────────────────
    stats = {
        "domain": args.domain,
        "documents": doc_count,
        "total_characters": total_chars,
        "filter_stats": quality_filter.stats.summary(),
    }
    if not args.skip_dedup:
        stats["dedup_stats"] = deduplicator.stats.summary()

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Pipeline stats saved to %s", stats_path)


def main():
    parser = argparse.ArgumentParser(description="Build a domain corpus")
    parser.add_argument("--domain", type=str, required=True, help="Domain name (e.g. legal)")
    parser.add_argument("--output", type=str, required=True, help="Output directory path")
    parser.add_argument("--source", type=str, help="HuggingFace dataset name to load")
    parser.add_argument("--subset", type=str, help="Dataset subset/config name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--text-field", type=str, default="text", help="Field containing text")
    parser.add_argument("--local-path", type=str, help="Path to local JSONL file")
    parser.add_argument("--max-docs", type=int, help="Maximum number of documents to load")
    parser.add_argument("--min-quality", type=float, default=0.4, help="Minimum quality score")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument(
        "--format", type=str, choices=["completion", "qa"],
        help="Format corpus for training (completion or qa mode)",
    )
    parser.add_argument("--max-length", type=int, default=8000, help="Max chars per training example")

    args = parser.parse_args()
    build_corpus(args)


if __name__ == "__main__":
    main()
