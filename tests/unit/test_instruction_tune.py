"""Tests for instruction tuning configuration and data loading."""

import json
from pathlib import Path

from src.adaptation.stages.instruction_tune import TrainingResult, _build_training_arguments
from src.config.settings import TrainingConfig


def _make_sft_jsonl(tmp_path: Path, n: int = 10) -> Path:
    """Create a chat-formatted JSONL file for SFT tests."""
    path = tmp_path / "sft_data.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            record = {
                "messages": [
                    {"role": "system", "content": "You are a legal assistant."},
                    {"role": "user", "content": f"What is contract law principle {i}?"},
                    {"role": "assistant", "content": f"Contract law principle {i} states that..."},
                ],
            }
            f.write(json.dumps(record) + "\n")
    return path


class TestTrainingResult:
    def test_summary(self):
        result = TrainingResult(
            output_dir="/tmp/output",
            adapter_dir="/tmp/output/adapter",
            merged_dir=None,
            train_loss=0.5432,
            train_steps=1000,
            train_samples=5000,
        )
        s = result.summary()
        assert s["train_loss"] == 0.5432
        assert s["train_steps"] == 1000
        assert s["train_samples"] == 5000
        assert s["merged_dir"] is None


class TestBuildTrainingArguments:
    def test_default_config(self):
        config = TrainingConfig()
        args = _build_training_arguments(config, "/tmp/output")
        assert args.output_dir == "/tmp/output"
        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 4
        assert args.gradient_accumulation_steps == 4
        assert args.learning_rate == 2e-4
        assert args.warmup_ratio == 0.05
        assert args.lr_scheduler_type == "cosine"

    def test_bfloat16_enabled(self):
        config = TrainingConfig()
        config.model.torch_dtype = "bfloat16"
        args = _build_training_arguments(config, "/tmp/output")
        assert args.bf16 is True
        assert args.fp16 is False

    def test_float16_enabled(self):
        config = TrainingConfig()
        config.model.torch_dtype = "float16"
        args = _build_training_arguments(config, "/tmp/output")
        assert args.bf16 is False
        assert args.fp16 is True

    def test_wandb_enabled(self):
        config = TrainingConfig(wandb_project="test-project")
        args = _build_training_arguments(config, "/tmp/output")
        assert "wandb" in args.report_to

    def test_wandb_disabled(self):
        config = TrainingConfig(wandb_project="")
        args = _build_training_arguments(config, "/tmp/output")
        assert "wandb" not in args.report_to

    def test_custom_params(self):
        config = TrainingConfig()
        config.training.num_epochs = 5
        config.training.batch_size = 8
        config.optimizer.learning_rate = 1e-5
        args = _build_training_arguments(config, "/tmp/output")
        assert args.num_train_epochs == 5
        assert args.per_device_train_batch_size == 8
        assert args.learning_rate == 1e-5

    def test_max_steps_negative_means_use_epochs(self):
        config = TrainingConfig()
        config.training.max_steps = -1
        args = _build_training_arguments(config, "/tmp/output")
        assert args.max_steps == -1

    def test_max_steps_positive(self):
        config = TrainingConfig()
        config.training.max_steps = 500
        args = _build_training_arguments(config, "/tmp/output")
        assert args.max_steps == 500


class TestLoadSftDataset:
    def test_load_from_jsonl(self, tmp_path):
        from src.adaptation.stages.instruction_tune import load_sft_dataset

        path = _make_sft_jsonl(tmp_path, n=5)
        ds = load_sft_dataset(path, tokenizer=None)
        assert len(ds) == 5
        assert "messages" in ds.column_names

    def test_messages_structure(self, tmp_path):
        from src.adaptation.stages.instruction_tune import load_sft_dataset

        path = _make_sft_jsonl(tmp_path, n=3)
        ds = load_sft_dataset(path, tokenizer=None)
        first = ds[0]
        assert isinstance(first["messages"], list)
        assert first["messages"][0]["role"] == "system"
        assert first["messages"][1]["role"] == "user"
        assert first["messages"][2]["role"] == "assistant"
