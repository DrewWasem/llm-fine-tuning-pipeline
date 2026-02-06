"""Supervised Fine-Tuning (SFT) stage using trl's SFTTrainer.

Handles the instruction tuning stage of domain adaptation:
loading chat-formatted data, configuring the trainer, running
training with W&B logging, and saving results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result from a training run."""

    output_dir: str
    adapter_dir: str | None
    merged_dir: str | None
    train_loss: float | None
    train_steps: int
    train_samples: int

    def summary(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "adapter_dir": self.adapter_dir,
            "merged_dir": self.merged_dir,
            "train_loss": self.train_loss,
            "train_steps": self.train_steps,
            "train_samples": self.train_samples,
        }


def _build_training_arguments(config: TrainingConfig, output_dir: str):
    """Build HuggingFace TrainingArguments from our config."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_steps=config.training.max_steps if config.training.max_steps > 0 else -1,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        warmup_ratio=config.optimizer.warmup_ratio,
        lr_scheduler_type=config.optimizer.lr_scheduler,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        save_total_limit=3,
        bf16=config.model.torch_dtype == "bfloat16",
        fp16=config.model.torch_dtype == "float16",
        gradient_checkpointing=config.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="wandb" if config.wandb_project else "none",
        run_name=f"{config.wandb_project}-sft",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )


def load_sft_dataset(data_path: str | Path, tokenizer, max_length: int = 2048):
    """Load a chat-formatted JSONL dataset for SFT training.

    Expects JSONL with {"messages": [{"role": ..., "content": ...}, ...]} format,
    as produced by DomainChatFormatter.

    Args:
        data_path: Path to JSONL file.
        tokenizer: HuggingFace tokenizer for applying chat templates.
        max_length: Maximum sequence length.

    Returns:
        HuggingFace Dataset ready for SFTTrainer.
    """
    from datasets import load_dataset

    logger.info("Loading SFT dataset from: %s", data_path)
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    logger.info("Dataset loaded: %d examples", len(dataset))
    return dataset


def _format_chat_for_sft(example: dict, tokenizer) -> str:
    """Apply the tokenizer's chat template to a messages list.

    Falls back to simple formatting if tokenizer has no chat template.
    """
    messages = example["messages"]

    # Check if tokenizer has a chat template
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Fallback: simple text formatting for tokenizers without chat templates
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
    return "".join(parts)


def run_sft(
    model,
    tokenizer,
    dataset,
    config: TrainingConfig,
    output_dir: str | Path | None = None,
    eval_dataset=None,
    resume_from_checkpoint: str | None = None,
) -> TrainingResult:
    """Run supervised fine-tuning with SFTTrainer.

    Args:
        model: Model (base or with LoRA adapters applied).
        tokenizer: HuggingFace tokenizer.
        dataset: Training dataset (HuggingFace Dataset with "messages" field).
        config: Training configuration.
        output_dir: Directory for checkpoints and outputs.
        eval_dataset: Optional evaluation dataset.
        resume_from_checkpoint: Path to resume from a checkpoint.

    Returns:
        TrainingResult with paths and metrics.
    """
    from trl import SFTConfig, SFTTrainer

    output_dir = str(output_dir or config.output_dir)

    logger.info("Configuring SFT trainer (output=%s)", output_dir)

    # Build SFTConfig (extends TrainingArguments)
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_steps=config.training.max_steps if config.training.max_steps > 0 else -1,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        warmup_ratio=config.optimizer.warmup_ratio,
        lr_scheduler_type=config.optimizer.lr_scheduler,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        bf16=config.model.torch_dtype == "bfloat16",
        fp16=config.model.torch_dtype == "float16",
        gradient_checkpointing=config.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="wandb" if config.wandb_project else "none",
        run_name=f"{config.wandb_project}-sft",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        max_length=config.data.max_length,
        packing=config.data.packing,
        dataset_text_field="text",
    )

    # Pre-format messages into text using chat template
    def format_fn(example):
        text = _format_chat_for_sft(example, tokenizer)
        return {"text": text}

    logger.info("Formatting dataset with chat template...")
    formatted_dataset = dataset.map(format_fn, remove_columns=["messages"])
    if "metadata" in formatted_dataset.column_names:
        formatted_dataset = formatted_dataset.remove_columns(["metadata"])
    if "doc_id" in formatted_dataset.column_names:
        formatted_dataset = formatted_dataset.remove_columns(["doc_id"])

    formatted_eval = None
    if eval_dataset is not None:
        formatted_eval = eval_dataset.map(format_fn, remove_columns=["messages"])
        if "metadata" in formatted_eval.column_names:
            formatted_eval = formatted_eval.remove_columns(["metadata"])
        if "doc_id" in formatted_eval.column_names:
            formatted_eval = formatted_eval.remove_columns(["doc_id"])

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=formatted_dataset,
        eval_dataset=formatted_eval,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training (%d examples)...", len(formatted_dataset))
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final adapter
    adapter_dir = str(Path(output_dir) / "final_adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Final adapter saved to: %s", adapter_dir)

    # Extract metrics
    metrics = train_result.metrics
    train_loss = metrics.get("train_loss")
    train_steps = metrics.get("train_steps", 0)

    logger.info("Training complete. Loss=%.4f, Steps=%d", train_loss or 0, train_steps)

    return TrainingResult(
        output_dir=output_dir,
        adapter_dir=adapter_dir,
        merged_dir=None,
        train_loss=train_loss,
        train_steps=train_steps,
        train_samples=len(formatted_dataset),
    )
