"""Model and tokenizer loading with quantization support.

Loads HuggingFace causal LM models with optional 4-bit quantization
via bitsandbytes, flash attention, and gradient checkpointing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import ModelConfig

logger = logging.getLogger(__name__)

# Map string dtype names to torch dtypes (resolved lazily)
_DTYPE_MAP = {
    "float32": "torch.float32",
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
}


@dataclass
class LoadedModel:
    """Container for a loaded model and its tokenizer."""

    model: object  # AutoModelForCausalLM
    tokenizer: object  # AutoTokenizer
    config: ModelConfig

    @property
    def model_name(self) -> str:
        return self.config.base_model

    @property
    def is_quantized(self) -> bool:
        return self.config.load_in_4bit


def _resolve_torch_dtype(dtype_str: str):
    """Convert a string dtype to a torch dtype object."""
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown torch dtype: {dtype_str}. Choose from: {list(mapping.keys())}")
    return mapping[dtype_str]


def _build_quantization_config(config: ModelConfig):
    """Build a BitsAndBytesConfig for 4-bit quantization."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_resolve_torch_dtype(config.torch_dtype),
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str, **kwargs):
    """Load a HuggingFace tokenizer.

    Args:
        model_name: HuggingFace model identifier or local path.
        **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained.

    Returns:
        AutoTokenizer instance with pad_token set.
    """
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token: %s", tokenizer.pad_token)

    return tokenizer


def load_model(config: ModelConfig, **kwargs) -> LoadedModel:
    """Load a model and tokenizer from a ModelConfig.

    Supports:
    - 4-bit quantization via bitsandbytes
    - Flash Attention 2
    - Gradient checkpointing
    - Loading from local paths or HuggingFace Hub

    Args:
        config: ModelConfig with base_model, torch_dtype, quantization settings.
        **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained.

    Returns:
        LoadedModel containing the model, tokenizer, and config.
    """
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(config.base_model)

    model_kwargs = {
        "torch_dtype": _resolve_torch_dtype(config.torch_dtype),
        "device_map": "auto",
        **kwargs,
    }

    # Quantization
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = _build_quantization_config(config)
        logger.info("4-bit quantization enabled (nf4, double quant)")

    # Flash Attention
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
        logger.info("Attention implementation: %s", config.attn_implementation)

    logger.info("Loading model: %s", config.base_model)
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)

    # Gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Log model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: %.2fB params (%.2fM trainable)",
        total_params / 1e9,
        trainable_params / 1e6,
    )

    return LoadedModel(model=model, tokenizer=tokenizer, config=config)


def load_model_for_inference(
    model_path: str | Path,
    base_model: str | None = None,
    load_in_4bit: bool = False,
):
    """Load a trained model (base + adapter) for inference.

    Args:
        model_path: Path to saved adapter or merged model.
        base_model: Base model name (required if model_path is an adapter).
        load_in_4bit: Whether to load in 4-bit quantization.

    Returns:
        LoadedModel for inference.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model_path = Path(model_path)

    # Check if this is a PEFT adapter (has adapter_config.json)
    is_adapter = (model_path / "adapter_config.json").exists()

    if is_adapter:
        if not base_model:
            raise ValueError("base_model is required when loading a PEFT adapter")

        config = ModelConfig(
            base_model=base_model,
            load_in_4bit=load_in_4bit,
            gradient_checkpointing=False,
        )
        loaded = load_model(config)
        logger.info("Loading PEFT adapter from: %s", model_path)
        loaded.model = PeftModel.from_pretrained(loaded.model, str(model_path))
        return loaded
    else:
        # Merged model — load directly
        config = ModelConfig(
            base_model=str(model_path),
            load_in_4bit=load_in_4bit,
            gradient_checkpointing=False,
        )
        return load_model(config)
