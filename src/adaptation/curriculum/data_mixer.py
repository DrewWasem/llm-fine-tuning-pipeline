"""Data mixing for domain adaptation training.

Mixes domain-specific data with general-purpose data at a configurable
ratio to prevent catastrophic forgetting during fine-tuning.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.settings import DataConfig

logger = logging.getLogger(__name__)


def load_dataset_from_path(path: str | Path, split: str = "train"):
    """Load a HuggingFace Dataset from a local JSONL file.

    Args:
        path: Path to JSONL file.
        split: Split name (default "train").

    Returns:
        HuggingFace Dataset.
    """
    from datasets import load_dataset

    return load_dataset("json", data_files=str(path), split=split)


def mix_datasets(
    domain_dataset,
    general_dataset=None,
    domain_weight: float = 0.8,
    general_weight: float = 0.2,
    seed: int = 42,
    max_samples: int | None = None,
):
    """Mix domain and general datasets at a specified ratio.

    The mixing is done by sampling from each dataset proportionally.
    This helps prevent catastrophic forgetting by keeping some general
    knowledge in the training data.

    Args:
        domain_dataset: HuggingFace Dataset of domain-specific examples.
        general_dataset: HuggingFace Dataset of general examples (optional).
        domain_weight: Proportion of domain data (0.0 to 1.0).
        general_weight: Proportion of general data (0.0 to 1.0).
        seed: Random seed for reproducible sampling.
        max_samples: Maximum total samples in the mixed dataset.

    Returns:
        HuggingFace Dataset with interleaved domain and general examples.
    """
    from datasets import concatenate_datasets

    if general_dataset is None or general_weight <= 0:
        logger.info("No general dataset provided, using domain-only (%d examples)", len(domain_dataset))
        if max_samples and len(domain_dataset) > max_samples:
            return domain_dataset.shuffle(seed=seed).select(range(max_samples))
        return domain_dataset.shuffle(seed=seed)

    # Normalize weights
    total_weight = domain_weight + general_weight
    domain_ratio = domain_weight / total_weight
    general_ratio = general_weight / total_weight

    # Determine sample counts
    if max_samples:
        total = max_samples
    else:
        # Scale to the domain dataset size
        total = int(len(domain_dataset) / domain_ratio)

    n_domain = min(int(total * domain_ratio), len(domain_dataset))
    n_general = min(int(total * general_ratio), len(general_dataset))

    logger.info(
        "Mixing: %d domain (%.0f%%) + %d general (%.0f%%) = %d total",
        n_domain,
        domain_ratio * 100,
        n_general,
        general_ratio * 100,
        n_domain + n_general,
    )

    # Sample from each dataset
    domain_sampled = domain_dataset.shuffle(seed=seed).select(range(n_domain))
    general_sampled = general_dataset.shuffle(seed=seed).select(range(n_general))

    # Concatenate and shuffle
    mixed = concatenate_datasets([domain_sampled, general_sampled])
    mixed = mixed.shuffle(seed=seed)

    logger.info("Mixed dataset: %d examples", len(mixed))
    return mixed


class DataMixer:
    """High-level data mixer for training pipeline integration.

    Args:
        config: DataConfig with domain_weight, general_weight, etc.
        seed: Random seed for reproducible mixing.
    """

    def __init__(self, config: DataConfig, seed: int = 42):
        self.config = config
        self.seed = seed

    def mix(
        self,
        domain_path: str | Path,
        general_path: str | Path | None = None,
        max_samples: int | None = None,
    ):
        """Load and mix datasets from file paths.

        Args:
            domain_path: Path to domain JSONL dataset.
            general_path: Path to general JSONL dataset (optional).
            max_samples: Maximum total examples.

        Returns:
            Mixed HuggingFace Dataset.
        """
        domain_ds = load_dataset_from_path(domain_path)
        general_ds = load_dataset_from_path(general_path) if general_path else None

        return mix_datasets(
            domain_dataset=domain_ds,
            general_dataset=general_ds,
            domain_weight=self.config.domain_weight,
            general_weight=self.config.general_weight,
            seed=self.seed,
            max_samples=max_samples,
        )
