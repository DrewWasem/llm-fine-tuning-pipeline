"""Tests for model loader configuration and utilities."""

import pytest

from src.config.settings import ModelConfig
from src.models.loader import _resolve_torch_dtype


class TestResolveTorchDtype:
    def test_bfloat16(self):
        import torch

        assert _resolve_torch_dtype("bfloat16") == torch.bfloat16

    def test_float16(self):
        import torch

        assert _resolve_torch_dtype("float16") == torch.float16

    def test_float32(self):
        import torch

        assert _resolve_torch_dtype("float32") == torch.float32

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown torch dtype"):
            _resolve_torch_dtype("int8")


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.base_model == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert config.torch_dtype == "bfloat16"
        assert config.load_in_4bit is True
        assert config.gradient_checkpointing is True

    def test_custom_config(self):
        config = ModelConfig(
            base_model="mistralai/Mistral-7B-v0.1",
            torch_dtype="float16",
            load_in_4bit=False,
        )
        assert config.base_model == "mistralai/Mistral-7B-v0.1"
        assert config.torch_dtype == "float16"
        assert config.load_in_4bit is False


class TestBuildQuantizationConfig:
    def test_creates_4bit_config(self):
        import torch

        from src.models.loader import _build_quantization_config

        config = ModelConfig()
        bnb_config = _build_quantization_config(config)
        assert bnb_config.load_in_4bit is True
        assert bnb_config.bnb_4bit_quant_type == "nf4"
        assert bnb_config.bnb_4bit_compute_dtype == torch.bfloat16
        assert bnb_config.bnb_4bit_use_double_quant is True


class TestLoadedModel:
    def test_loaded_model_properties(self):
        from src.models.loader import LoadedModel

        config = ModelConfig(base_model="test-model", load_in_4bit=True)
        loaded = LoadedModel(model=None, tokenizer=None, config=config)
        assert loaded.model_name == "test-model"
        assert loaded.is_quantized is True

    def test_loaded_model_not_quantized(self):
        from src.models.loader import LoadedModel

        config = ModelConfig(base_model="test-model", load_in_4bit=False)
        loaded = LoadedModel(model=None, tokenizer=None, config=config)
        assert loaded.is_quantized is False
