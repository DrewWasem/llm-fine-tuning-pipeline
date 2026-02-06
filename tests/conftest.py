"""Shared test fixtures for the domain adaptation toolkit."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    return project_root / "configs"


@pytest.fixture
def legal_domain_config(configs_dir: Path) -> Path:
    return configs_dir / "domains" / "legal.yaml"


@pytest.fixture
def lora_training_config(configs_dir: Path) -> Path:
    return configs_dir / "training" / "lora_adapt.yaml"
