"""LoRA adapter setup for domain adaptation.

Wraps PEFT's LoRA configuration and applies it to a loaded model.
Supports saving, loading, and merging adapters back into the base model.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.settings import LoRAConfig

logger = logging.getLogger(__name__)


def create_lora_config(config: LoRAConfig):
    """Create a PEFT LoraConfig from our config model.

    Args:
        config: LoRAConfig with rank, alpha, dropout, target modules, etc.

    Returns:
        peft.LoraConfig instance.
    """
    from peft import LoraConfig, TaskType

    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    }

    task_type = task_type_map.get(config.task_type, TaskType.CAUSAL_LM)

    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=task_type,
    )

    logger.info(
        "LoRA config: r=%d, alpha=%d, dropout=%.2f, targets=%s",
        config.r,
        config.lora_alpha,
        config.lora_dropout,
        config.target_modules,
    )
    return lora_config


def apply_lora(model, lora_config) -> object:
    """Apply LoRA adapters to a model.

    Args:
        model: A HuggingFace model (AutoModelForCausalLM).
        lora_config: A peft.LoraConfig instance.

    Returns:
        PeftModel with LoRA adapters applied.
    """
    from peft import get_peft_model, prepare_model_for_kbit_training

    # Prepare for quantized training if the model is quantized
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)
        logger.info("Prepared model for k-bit training")

    peft_model = get_peft_model(model, lora_config)

    # Log trainable params
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    pct = 100 * trainable_params / total_params if total_params > 0 else 0

    logger.info(
        "LoRA applied: %dM trainable / %dM total (%.2f%%)",
        trainable_params // 1_000_000,
        total_params // 1_000_000,
        pct,
    )
    return peft_model


def save_adapter(model, output_dir: str | Path) -> Path:
    """Save LoRA adapter weights to disk.

    Args:
        model: PeftModel with trained LoRA adapters.
        output_dir: Directory to save the adapter to.

    Returns:
        Path to the saved adapter directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    logger.info("Adapter saved to: %s", output_dir)
    return output_dir


def merge_and_save(model, tokenizer, output_dir: str | Path) -> Path:
    """Merge LoRA adapters into the base model and save.

    This produces a standalone model that doesn't require PEFT at inference time.

    Args:
        model: PeftModel with trained LoRA adapters.
        tokenizer: The associated tokenizer.
        output_dir: Directory to save the merged model to.

    Returns:
        Path to the saved merged model directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Merged model saved to: %s", output_dir)
    return output_dir
