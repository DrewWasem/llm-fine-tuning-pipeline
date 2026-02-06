#!/usr/bin/env python3
"""Run the full domain adaptation pipeline end-to-end.

Orchestrates: corpus building → synthetic data generation → training → evaluation.

Usage:
    # Full pipeline with default settings
    python scripts/run_pipeline.py --domain legal --output-dir outputs/legal

    # Skip corpus building (use existing corpus)
    python scripts/run_pipeline.py \
        --domain legal \
        --output-dir outputs/legal \
        --corpus data/processed/legal_corpus/corpus.jsonl \
        --skip-corpus

    # Skip training (evaluate existing model)
    python scripts/run_pipeline.py \
        --domain legal \
        --output-dir outputs/legal \
        --model models/legal-lora/final_adapter \
        --skip-training

    # Dry run (show what would be executed)
    python scripts/run_pipeline.py --domain legal --output-dir outputs/legal --dry-run

    # Quick test with limited data
    python scripts/run_pipeline.py \
        --domain legal \
        --output-dir outputs/legal-test \
        --max-docs 100 \
        --max-samples 500 \
        --max-steps 50
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Results from a pipeline run."""

    domain: str
    output_dir: str
    started_at: str
    completed_at: str = ""
    stages_completed: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)
    corpus_path: str = ""
    training_data_path: str = ""
    model_path: str = ""
    eval_report_path: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success": self.success,
            "stages_completed": self.stages_completed,
            "stages_skipped": self.stages_skipped,
            "corpus_path": self.corpus_path,
            "training_data_path": self.training_data_path,
            "model_path": self.model_path,
            "eval_report_path": self.eval_report_path,
            "errors": self.errors,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def run_command(cmd: list[str], dry_run: bool = False) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    cmd_str = " ".join(cmd)
    logger.info("Running: %s", cmd_str)

    if dry_run:
        logger.info("[DRY RUN] Would execute: %s", cmd_str)
        return True, ""

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        if result.returncode != 0:
            logger.error("Command failed: %s", result.stderr)
            return False, result.stderr
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 2 hours"
    except Exception as e:
        return False, str(e)


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    """Execute the full pipeline."""
    from src.config.settings import load_domain_config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = PipelineResult(
        domain=args.domain,
        output_dir=str(output_dir),
        started_at=datetime.now().isoformat(),
    )

    domain_cfg = load_domain_config(args.domain)
    logger.info("=" * 60)
    logger.info("Starting pipeline for domain: %s", domain_cfg.name.value)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 60)

    # ── Stage 1: Build Corpus ────────────────────────────────────────────────
    corpus_dir = output_dir / "corpus"
    corpus_path = corpus_dir / "corpus.jsonl"
    formatted_path = corpus_dir / "formatted.jsonl"

    if args.skip_corpus and args.corpus:
        logger.info("Skipping corpus building (using provided corpus)")
        result.stages_skipped.append("corpus")
        result.corpus_path = args.corpus
        corpus_path = Path(args.corpus)
        formatted_path = corpus_path.parent / "formatted.jsonl"
    elif args.skip_corpus:
        logger.info("Skipping corpus building")
        result.stages_skipped.append("corpus")
    else:
        logger.info("\n[Stage 1/4] Building corpus...")
        cmd = [
            sys.executable, "scripts/build_corpus.py",
            "--domain", args.domain,
            "--output", str(corpus_dir),
            "--format", "completion",
        ]
        if args.source:
            cmd.extend(["--source", args.source])
        if args.max_docs:
            cmd.extend(["--max-docs", str(args.max_docs)])

        success, error = run_command(cmd, args.dry_run)
        if not success:
            result.errors.append(f"Corpus building failed: {error}")
            result.completed_at = datetime.now().isoformat()
            return result

        result.stages_completed.append("corpus")
        result.corpus_path = str(corpus_path)

    # ── Stage 2: Generate Synthetic Data ─────────────────────────────────────
    synthetic_path = output_dir / "synthetic" / "training_data.jsonl"

    if args.skip_synthetic:
        logger.info("Skipping synthetic data generation")
        result.stages_skipped.append("synthetic")
        training_data = formatted_path if formatted_path.exists() else corpus_path
    else:
        logger.info("\n[Stage 2/4] Generating synthetic training data...")
        cmd = [
            sys.executable, "scripts/generate_synthetic.py",
            "--domain", args.domain,
            "--corpus", str(corpus_path),
            "--output", str(synthetic_path),
        ]
        if args.max_samples:
            cmd.extend(["--num-samples", str(args.max_samples)])

        success, error = run_command(cmd, args.dry_run)
        if not success:
            result.errors.append(f"Synthetic generation failed: {error}")
            result.completed_at = datetime.now().isoformat()
            return result

        result.stages_completed.append("synthetic")
        training_data = synthetic_path

    result.training_data_path = str(training_data)

    # ── Stage 3: Train Model ─────────────────────────────────────────────────
    model_dir = output_dir / "model"

    if args.skip_training and args.model:
        logger.info("Skipping training (using provided model)")
        result.stages_skipped.append("training")
        result.model_path = args.model
    elif args.skip_training:
        logger.info("Skipping training")
        result.stages_skipped.append("training")
    else:
        logger.info("\n[Stage 3/4] Training model...")
        cmd = [
            sys.executable, "scripts/train_domain_model.py",
            "--domain", args.domain,
            "--data", str(training_data),
            "--output", str(model_dir),
        ]
        if args.config:
            cmd.extend(["--config", args.config])
        if args.base_model:
            cmd.extend(["--base-model", args.base_model])
        if args.max_steps:
            cmd.extend(["--max-steps", str(args.max_steps)])

        success, error = run_command(cmd, args.dry_run)
        if not success:
            result.errors.append(f"Training failed: {error}")
            result.completed_at = datetime.now().isoformat()
            return result

        result.stages_completed.append("training")
        result.model_path = str(model_dir / "final_adapter")

    # ── Stage 4: Evaluate Model ──────────────────────────────────────────────
    eval_report = output_dir / "evaluation" / "report.json"

    if args.skip_eval:
        logger.info("Skipping evaluation")
        result.stages_skipped.append("evaluation")
    elif not result.model_path:
        logger.warning("No model to evaluate, skipping evaluation")
        result.stages_skipped.append("evaluation")
    else:
        logger.info("\n[Stage 4/4] Evaluating model...")
        eval_report.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "scripts/evaluate_domain.py",
            "--model", result.model_path,
            "--domain", args.domain,
            "--output", str(eval_report),
        ]
        if args.base_model:
            cmd.extend(["--base-model", args.base_model])

        success, error = run_command(cmd, args.dry_run)
        if not success:
            result.errors.append(f"Evaluation failed: {error}")
            result.completed_at = datetime.now().isoformat()
            return result

        result.stages_completed.append("evaluation")
        result.eval_report_path = str(eval_report)

    # ── Finalize ─────────────────────────────────────────────────────────────
    result.completed_at = datetime.now().isoformat()

    # Save pipeline result
    result_path = output_dir / "pipeline_result.json"
    result.save(result_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Domain: {result.domain}")
    print(f"Success: {result.success}")
    print(f"Stages completed: {', '.join(result.stages_completed) or 'none'}")
    print(f"Stages skipped: {', '.join(result.stages_skipped) or 'none'}")
    if result.corpus_path:
        print(f"Corpus: {result.corpus_path}")
    if result.training_data_path:
        print(f"Training data: {result.training_data_path}")
    if result.model_path:
        print(f"Model: {result.model_path}")
    if result.eval_report_path:
        print(f"Eval report: {result.eval_report_path}")
    if result.errors:
        print(f"Errors: {result.errors}")
    print(f"Full result: {result_path}")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the full domain adaptation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--domain", type=str, required=True, help="Domain name (e.g., legal)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for all artifacts")

    # Data sources
    parser.add_argument("--source", type=str, help="HuggingFace dataset for corpus")
    parser.add_argument("--corpus", type=str, help="Path to existing corpus JSONL")

    # Model
    parser.add_argument("--base-model", type=str, help="Base model name or path")
    parser.add_argument("--model", type=str, help="Path to existing trained model")
    parser.add_argument("--config", type=str, help="Training config YAML path")

    # Limits
    parser.add_argument("--max-docs", type=int, help="Max documents for corpus")
    parser.add_argument("--max-samples", type=int, help="Max synthetic training samples")
    parser.add_argument("--max-steps", type=int, help="Max training steps")

    # Skip stages
    parser.add_argument("--skip-corpus", action="store_true", help="Skip corpus building")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")

    # Execution
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()
    result = run_pipeline(args)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
