#!/usr/bin/env python3
"""Extract domain terminology from a corpus.

Usage:
    python scripts/extract_terminology.py \
        --corpus data/processed/legal_corpus/corpus.jsonl \
        --output data/terminology/legal_terms.json

    python scripts/extract_terminology.py \
        --corpus data/processed/legal_corpus/corpus.jsonl \
        --output data/terminology/legal_terms.json \
        --no-spacy --min-frequency 50 --top-k 1000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import TerminologyConfig, load_domain_config
from src.corpus.terminology.extractor import TerminologyExtractor
from src.data.loaders.base_loader import JSONLLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract domain terminology from a corpus")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--domain", type=str, help="Domain name (loads config for terminology settings)")
    parser.add_argument("--min-frequency", type=int, help="Minimum term frequency (overrides config)")
    parser.add_argument("--top-k", type=int, default=5000, help="Maximum number of terms")
    parser.add_argument("--max-ngram", type=int, default=3, help="Maximum n-gram size")
    parser.add_argument("--no-spacy", action="store_true", help="Disable spaCy NER (TF-IDF only)")
    parser.add_argument("--max-docs", type=int, help="Limit number of documents to process")

    args = parser.parse_args()

    # Load terminology config from domain if provided
    if args.domain:
        domain_cfg = load_domain_config(args.domain)
        term_config = domain_cfg.terminology
    else:
        term_config = TerminologyConfig()

    # Override with CLI args
    if args.min_frequency is not None:
        term_config = TerminologyConfig(
            method=term_config.method,
            min_frequency=args.min_frequency,
            pos_tags=term_config.pos_tags,
        )

    # Load corpus
    logger.info("Loading corpus from %s", args.corpus)
    loader = JSONLLoader(path=args.corpus, source_name="corpus")
    documents = loader.load_all()

    if args.max_docs:
        documents = documents[: args.max_docs]

    logger.info("Loaded %d documents", len(documents))

    # Extract terminology
    extractor = TerminologyExtractor(
        config=term_config,
        use_spacy=not args.no_spacy,
        max_ngram=args.max_ngram,
        top_k=args.top_k,
    )

    terms = extractor.extract(documents)
    logger.info("Extracted %d terms", len(terms))

    # Save
    extractor.save(terms, args.output)
    logger.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
