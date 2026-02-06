"""Tests for data mixing."""

import json
from pathlib import Path

from datasets import Dataset

from src.adaptation.curriculum.data_mixer import DataMixer, mix_datasets
from src.config.settings import DataConfig


def _make_jsonl_dataset(tmp_path: Path, name: str, n: int, prefix: str = "") -> Path:
    """Create a JSONL file with n chat examples."""
    path = tmp_path / f"{name}.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            record = {
                "messages": [
                    {"role": "user", "content": f"{prefix}Question {i}"},
                    {"role": "assistant", "content": f"{prefix}Answer {i}"},
                ],
            }
            f.write(json.dumps(record) + "\n")
    return path


def _make_hf_dataset(n: int, prefix: str = "") -> Dataset:
    """Create an in-memory HF Dataset with n examples."""
    return Dataset.from_dict({
        "messages": [
            [
                {"role": "user", "content": f"{prefix}Q{i}"},
                {"role": "assistant", "content": f"{prefix}A{i}"},
            ]
            for i in range(n)
        ],
    })


class TestMixDatasets:
    def test_domain_only(self):
        domain_ds = _make_hf_dataset(100, prefix="legal ")
        mixed = mix_datasets(domain_ds, general_dataset=None, domain_weight=1.0, general_weight=0.0)
        assert len(mixed) == 100

    def test_80_20_mix(self):
        domain_ds = _make_hf_dataset(80, prefix="legal ")
        general_ds = _make_hf_dataset(100, prefix="general ")
        mixed = mix_datasets(
            domain_ds,
            general_ds,
            domain_weight=0.8,
            general_weight=0.2,
            max_samples=100,
        )
        assert len(mixed) == 100

    def test_max_samples_cap(self):
        domain_ds = _make_hf_dataset(500, prefix="legal ")
        mixed = mix_datasets(domain_ds, max_samples=50)
        assert len(mixed) == 50

    def test_50_50_mix(self):
        domain_ds = _make_hf_dataset(100, prefix="legal ")
        general_ds = _make_hf_dataset(100, prefix="general ")
        mixed = mix_datasets(
            domain_ds,
            general_ds,
            domain_weight=0.5,
            general_weight=0.5,
            max_samples=100,
        )
        assert len(mixed) == 100

    def test_general_weight_zero_ignores_general(self):
        domain_ds = _make_hf_dataset(50, prefix="legal ")
        general_ds = _make_hf_dataset(50, prefix="general ")
        mixed = mix_datasets(
            domain_ds,
            general_ds,
            domain_weight=1.0,
            general_weight=0.0,
        )
        assert len(mixed) == 50

    def test_seed_reproducibility(self):
        domain_ds = _make_hf_dataset(100)
        general_ds = _make_hf_dataset(100)
        mixed1 = mix_datasets(domain_ds, general_ds, seed=42, max_samples=50)
        mixed2 = mix_datasets(domain_ds, general_ds, seed=42, max_samples=50)
        assert mixed1["messages"] == mixed2["messages"]


class TestDataMixer:
    def test_mix_from_files(self, tmp_path):
        domain_path = _make_jsonl_dataset(tmp_path, "domain", 50, prefix="legal ")
        general_path = _make_jsonl_dataset(tmp_path, "general", 50, prefix="general ")

        config = DataConfig(domain_weight=0.8, general_weight=0.2)
        mixer = DataMixer(config=config)
        mixed = mixer.mix(domain_path, general_path, max_samples=40)
        assert len(mixed) == 40

    def test_mix_domain_only(self, tmp_path):
        domain_path = _make_jsonl_dataset(tmp_path, "domain", 30)

        config = DataConfig(domain_weight=1.0, general_weight=0.0)
        mixer = DataMixer(config=config)
        mixed = mixer.mix(domain_path)
        assert len(mixed) == 30
