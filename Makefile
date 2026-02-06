.PHONY: install install-dev lint format test test-unit test-integration clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,eval]"
	pre-commit install

lint:
	ruff check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

format:
	ruff check --fix src/ tests/ scripts/
	ruff format src/ tests/ scripts/

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info .mypy_cache .ruff_cache .pytest_cache

# Domain pipeline commands
corpus:
	python scripts/build_corpus.py --domain $(domain) --output data/processed/$(domain)_corpus

terminology:
	python scripts/extract_terminology.py --corpus data/processed/$(domain)_corpus --output data/terminology/$(domain)_terms.json

synthetic:
	python scripts/generate_synthetic.py --domain $(domain) --corpus data/processed/$(domain)_corpus/corpus.jsonl --output data/processed/$(domain)_synthetic.jsonl

train:
	python scripts/train_domain_model.py --domain $(domain) --config configs/training/lora_adapt.yaml

evaluate:
	python scripts/evaluate_domain.py --model models/$(domain)-lora --domain $(domain)

# Full end-to-end pipeline
pipeline:
	python scripts/run_pipeline.py --domain $(domain) --output-dir outputs/$(domain)

# Quick test pipeline (limited data for testing)
pipeline-test:
	python scripts/run_pipeline.py --domain $(domain) --output-dir outputs/$(domain)-test --max-docs 100 --max-samples 500 --max-steps 50

# Dry run (show what would be executed)
pipeline-dry:
	python scripts/run_pipeline.py --domain $(domain) --output-dir outputs/$(domain) --dry-run
