#!/usr/bin/env python3
"""Train a domain-adapted model using LoRA + SFT.

Orchestrates: load config → load model → apply LoRA → load data → train → save.

Usage:
    # LoRA fine-tuning with default config
    python scripts/train_domain_model.py \
        --domain legal \
        --data data/processed/legal_corpus/formatted.jsonl \
        --output models/legal-lora

    # With custom training config
    python scripts/train_domain_model.py \
        --domain legal \
        --config configs/training/lora_adapt.yaml \
        --data data/processed/legal_corpus/formatted.jsonl \
        --output models/legal-lora

    # With general data mixing (prevent catastrophic forgetting)
    python scripts/train_domain_model.py \
        --domain legal \
        --data data/processed/legal_corpus/formatted.jsonl \
        --general-data data/processed/general/formatted.jsonl \
        --output models/legal-lora

    # Resume from checkpoint
    python scripts/train_domain_model.py \
        --domain legal \
        --data data/processed/legal_corpus/formatted.jsonl \
        --output models/legal-lora \
        --resume-from models/legal-lora/checkpoint-1000

    # Merge adapter into base model after training
    python scripts/train_domain_model.py \
        --domain legal \
        --data data/processed/legal_corpus/formatted.jsonl \
        --output models/legal-lora \
        --merge
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


def train(args: argparse.Namespace) -> None:
    # Lazy imports to keep CLI startup fast and allow --help without GPU
    from src.adaptation.curriculum.data_mixer import DataMixer
    from src.adaptation.stages.instruction_tune import load_sft_dataset, run_sft
    from src.adaptation.strategies.lora_adapt import (
        apply_lora,
        create_lora_config,
        merge_and_save,
    )
    from src.config.settings import TrainingConfig, load_training_config
    from src.models.loader import load_model

    # ── Step 1: Load config ─────────────────────────────────────────────────
    if args.config:
        config = load_training_config(args.config)
        logger.info("Loaded training config from: %s", args.config)
    else:
        config = TrainingConfig()
        logger.info("Using default training config")

    # Override output dir from CLI
    if args.output:
        config.output_dir = args.output

    # Override base model from CLI
    if args.base_model:
        config.model.base_model = args.base_model

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config for reproducibility
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)
    logger.info("Resolved config saved to: %s", config_path)

    # ── Step 2: Load model ──────────────────────────────────────────────────
    logger.info("Loading base model: %s", config.model.base_model)
    loaded = load_model(config.model)
    model = loaded.model
    tokenizer = loaded.tokenizer

    # ── Step 3: Apply LoRA ──────────────────────────────────────────────────
    lora_config = create_lora_config(config.lora)
    model = apply_lora(model, lora_config)

    # ── Step 4: Load and mix data ───────────────────────────────────────────
    if args.general_data:
        logger.info("Mixing domain + general data")
        mixer = DataMixer(config=config.data)
        dataset = mixer.mix(
            domain_path=args.data,
            general_path=args.general_data,
            max_samples=args.max_samples,
        )
    else:
        dataset = load_sft_dataset(args.data, tokenizer, max_length=config.data.max_length)
        if args.max_samples and len(dataset) > args.max_samples:
            dataset = dataset.shuffle(seed=42).select(range(args.max_samples))

    # Optional eval split
    eval_dataset = None
    if args.eval_data:
        eval_dataset = load_sft_dataset(
            args.eval_data, tokenizer, max_length=config.data.max_length
        )

    # ── Step 5: Train ───────────────────────────────────────────────────────
    result = run_sft(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        output_dir=str(output_dir),
        eval_dataset=eval_dataset,
        resume_from_checkpoint=args.resume_from,
    )

    logger.info("Training result: %s", result.summary())

    # ── Step 6: Optionally merge adapter ────────────────────────────────────
    if args.merge:
        merged_dir = output_dir / "merged"
        merge_and_save(model, tokenizer, merged_dir)
        result.merged_dir = str(merged_dir)

    # Save training summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result.summary(), f, indent=2)
    logger.info("Training summary saved to: %s", summary_path)
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Train a domain-adapted model using LoRA + SFT",
    )
    parser.add_argument(
        "--domain", type=str, required=True,
        help="Domain name (e.g. legal, medical)",
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to training config YAML (default: use built-in defaults)",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to training data JSONL (chat-formatted)",
    )
    parser.add_argument(
        "--general-data", type=str,
        help="Path to general data JSONL for mixing (prevents catastrophic forgetting)",
    )
    parser.add_argument(
        "--eval-data", type=str,
        help="Path to evaluation data JSONL",
    )
    parser.add_argument(
        "--base-model", type=str,
        help="Override base model name from config",
    )
    parser.add_argument(
        "--output", type=str,
        help="Output directory (overrides config output_dir)",
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Maximum training samples (for quick experiments)",
    )
    parser.add_argument(
        "--resume-from", type=str,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge LoRA adapter into base model after training",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
