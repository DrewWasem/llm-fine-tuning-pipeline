"""Tests for configuration loading and validation."""

from pathlib import Path

from src.config.settings import (
    DomainConfig,
    DomainName,
    TrainingConfig,
    TrainingStage,
    load_domain_config,
    load_pipeline_config,
    load_training_config,
    load_yaml,
)


class TestYAMLLoading:
    def test_load_yaml_file(self, legal_domain_config: Path):
        raw = load_yaml(legal_domain_config)
        assert isinstance(raw, dict)
        assert raw["name"] == "legal"

    def test_load_yaml_relative_path(self):
        raw = load_yaml("configs/domains/legal.yaml")
        assert raw["name"] == "legal"


class TestDomainConfig:
    def test_load_legal_domain(self):
        cfg = load_domain_config("legal")
        assert cfg.name == DomainName.LEGAL
        assert cfg.description != ""
        assert len(cfg.corpus.sources) > 0
        assert cfg.corpus.quality_filters.min_length == 500

    def test_load_legal_terminology(self):
        cfg = load_domain_config("legal")
        assert cfg.terminology.method == "tfidf_ner_hybrid"
        assert cfg.terminology.min_frequency == 100

    def test_load_legal_evaluation(self):
        cfg = load_domain_config("legal")
        assert len(cfg.evaluation.benchmarks) == 3
        names = [b.name for b in cfg.evaluation.benchmarks]
        assert "bar_exam" in names
        assert "contract_analysis" in names

    def test_domain_config_from_dict(self):
        cfg = DomainConfig(name="legal", description="test")
        assert cfg.name == DomainName.LEGAL


class TestTrainingConfig:
    def test_load_lora_config(self, lora_training_config: Path):
        cfg = load_training_config(lora_training_config)
        assert cfg.stage == TrainingStage.LORA
        assert cfg.lora.r == 16
        assert cfg.lora.lora_alpha == 32
        assert cfg.model.load_in_4bit is True

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        assert cfg.stage == TrainingStage.LORA
        assert cfg.optimizer.learning_rate == 2e-4

    def test_lora_target_modules(self, lora_training_config: Path):
        cfg = load_training_config(lora_training_config)
        assert "q_proj" in cfg.lora.target_modules
        assert "v_proj" in cfg.lora.target_modules


class TestPipelineConfig:
    def test_load_full_pipeline(self, lora_training_config: Path):
        cfg = load_pipeline_config("legal", lora_training_config)
        assert cfg.domain.name == DomainName.LEGAL
        assert cfg.training.stage == TrainingStage.LORA

    def test_pipeline_with_defaults(self):
        cfg = load_pipeline_config("legal")
        assert cfg.domain.name == DomainName.LEGAL
        assert cfg.training.stage == TrainingStage.LORA  # default
