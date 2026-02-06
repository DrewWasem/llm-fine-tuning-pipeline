"""Tests for LoRA adapter configuration and setup."""

from src.config.settings import LoRAConfig


class TestCreateLoraConfig:
    def test_default_lora_config(self):
        from src.adaptation.strategies.lora_adapt import create_lora_config

        config = LoRAConfig()
        lora_config = create_lora_config(config)

        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
        assert lora_config.lora_dropout == 0.05
        assert "q_proj" in lora_config.target_modules
        assert "v_proj" in lora_config.target_modules

    def test_custom_lora_config(self):
        from src.adaptation.strategies.lora_adapt import create_lora_config

        config = LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="all",
        )
        lora_config = create_lora_config(config)

        assert lora_config.r == 8
        assert lora_config.lora_alpha == 16
        assert lora_config.lora_dropout == 0.1
        assert lora_config.bias == "all"

    def test_causal_lm_task_type(self):
        from peft import TaskType

        from src.adaptation.strategies.lora_adapt import create_lora_config

        config = LoRAConfig(task_type="CAUSAL_LM")
        lora_config = create_lora_config(config)
        assert lora_config.task_type == TaskType.CAUSAL_LM

    def test_target_modules_list(self):
        from src.adaptation.strategies.lora_adapt import create_lora_config

        config = LoRAConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"])
        lora_config = create_lora_config(config)
        assert len(lora_config.target_modules) == 5
        assert "gate_proj" in lora_config.target_modules


class TestLoRAConfigModel:
    def test_defaults(self):
        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.task_type == "CAUSAL_LM"

    def test_from_dict(self):
        config = LoRAConfig(**{"r": 32, "lora_alpha": 64})
        assert config.r == 32
        assert config.lora_alpha == 64
