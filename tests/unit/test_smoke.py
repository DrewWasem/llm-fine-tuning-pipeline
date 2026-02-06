"""Smoke tests to verify the package is importable."""

import importlib


def test_import_config():
    from src.config import settings

    assert hasattr(settings, "load_domain_config")
    assert hasattr(settings, "load_training_config")
    assert hasattr(settings, "load_pipeline_config")


def test_import_top_level_packages():
    packages = [
        "src.corpus",
        "src.adaptation",
        "src.evaluation",
        "src.data",
        "src.models",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None
