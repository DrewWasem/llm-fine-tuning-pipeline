"""Pydantic-based configuration models for domain adaptation pipeline.

Loads and validates YAML configuration files for domains, training, and evaluation.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

# ── Enums ──────────────────────────────────────────────────────────────────────


class DomainName(str, Enum):
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    SCIENTIFIC = "scientific"
    CODE = "code"


class TrainingStage(str, Enum):
    PRETRAIN = "pretrain"
    INSTRUCTION = "instruction"
    ALIGN = "align"
    LORA = "lora"


class AdaptationStrategy(str, Enum):
    FULL_FINETUNE = "full_finetune"
    LORA = "lora"
    DOMAIN_LORA = "domain_lora"
    VOCAB_EXTENSION = "vocab_extension"


# ── Corpus Config ──────────────────────────────────────────────────────────────


class CorpusSource(BaseModel):
    name: str
    type: str
    categories: list[str] = Field(default_factory=list)
    date_range: list[str] = Field(default_factory=list)


class QualityFilterConfig(BaseModel):
    min_length: int = 500
    max_length: int = 50000
    language: str = "en"
    dedup_threshold: float = 0.85


class TerminologyConfig(BaseModel):
    method: str = "tfidf_ner_hybrid"
    min_frequency: int = 100
    pos_tags: list[str] = Field(default_factory=lambda: ["NOUN", "PROPN"])


class CorpusConfig(BaseModel):
    sources: list[CorpusSource] = Field(default_factory=list)
    quality_filters: QualityFilterConfig = Field(default_factory=QualityFilterConfig)


# ── Training Config ────────────────────────────────────────────────────────────


class OptimizerConfig(BaseModel):
    name: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class ModelConfig(BaseModel):
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = True
    load_in_4bit: bool = True


class DataConfig(BaseModel):
    domain_weight: float = 0.8
    general_weight: float = 0.2
    max_length: int = 2048
    packing: bool = True


class TrainingParams(BaseModel):
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = -1
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 500


class TrainingConfig(BaseModel):
    stage: TrainingStage = TrainingStage.LORA
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingParams = Field(default_factory=TrainingParams)
    output_dir: str = "models/output"
    wandb_project: str = "domain-adapt"


# ── Evaluation Config ──────────────────────────────────────────────────────────


class BenchmarkConfig(BaseModel):
    name: str
    weight: float = 1.0
    passing_threshold: float = 0.7


class RetentionConfig(BaseModel):
    benchmarks: list[str] = Field(
        default_factory=lambda: ["mmlu", "hellaswag"]
    )
    max_drop: float = 0.05


class EvaluationConfig(BaseModel):
    benchmarks: list[BenchmarkConfig] = Field(default_factory=list)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)


# ── Domain Config (top-level) ─────────────────────────────────────────────────


class DomainConfig(BaseModel):
    """Top-level domain configuration loaded from configs/domains/<domain>.yaml."""

    name: DomainName
    description: str = ""
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    terminology: TerminologyConfig = Field(default_factory=TerminologyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ── Pipeline Config ────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Complete pipeline configuration combining domain + training + evaluation."""

    domain: DomainConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @field_validator("domain", mode="before")
    @classmethod
    def parse_domain(cls, v: Any) -> Any:
        if isinstance(v, str):
            return DomainConfig(name=v)
        return v


# ── Config Loading ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_domain_config(domain: str | Path) -> DomainConfig:
    """Load a domain config by name or path.

    Args:
        domain: Either a domain name (e.g. "legal") which resolves to
                configs/domains/legal.yaml, or a direct path to a YAML file.
    """
    path = Path(domain)
    if not path.suffix:
        path = PROJECT_ROOT / "configs" / "domains" / f"{domain}.yaml"
    raw = load_yaml(path)
    # Support nested structure: {domain: {name: ...}, corpus: ...}
    # or flat structure: {name: ..., corpus: ...}
    if "domain" in raw and isinstance(raw["domain"], dict):
        merged = {**raw["domain"]}
        for key in ("corpus", "terminology", "evaluation"):
            if key in raw:
                merged[key] = raw[key]
        return DomainConfig(**merged)
    return DomainConfig(**raw)


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a training configuration from a YAML file."""
    raw = load_yaml(path)
    return TrainingConfig(**raw)


def load_pipeline_config(
    domain: str | Path,
    training: str | Path | None = None,
) -> PipelineConfig:
    """Load a complete pipeline config from domain + training YAML files.

    Args:
        domain: Domain name or path to domain YAML.
        training: Path to training YAML. If None, uses defaults.
    """
    domain_cfg = load_domain_config(domain)
    training_cfg = load_training_config(training) if training else TrainingConfig()
    return PipelineConfig(domain=domain_cfg, training=training_cfg)
