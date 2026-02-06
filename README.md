# Domain Adaptation Toolkit

A specialized LLM fine-tuning framework for adapting foundation models to professional domains (legal, medical, financial, scientific).

## Features

- **Domain Corpus Building**: Automated document ingestion with quality filtering and deduplication
- **Terminology Extraction**: TF-IDF and NER-based domain term extraction
- **Synthetic Data Generation**: Q&A pairs and instruction examples from domain documents
- **LoRA Fine-Tuning**: Parameter-efficient adaptation with 4-bit quantization
- **Domain Evaluation**: Benchmarks for domain knowledge and retention testing
- **End-to-End Pipeline**: Single command to run corpus → synthetic → train → evaluate

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DrewWasem/llm-fine-tuning-pipeline.git
cd llm-fine-tuning-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package
pip install -e ".[dev]"
```

### Run the Full Pipeline

```bash
# Full pipeline for legal domain
make pipeline domain=legal

# Or run directly
python scripts/run_pipeline.py --domain legal --output-dir outputs/legal
```

### Quick Test (Limited Data)

```bash
# Test with minimal data
make pipeline-test domain=legal

# Or with explicit limits
python scripts/run_pipeline.py \
    --domain legal \
    --output-dir outputs/legal-test \
    --max-docs 100 \
    --max-samples 500 \
    --max-steps 50
```

### Dry Run

```bash
# Preview what would be executed
make pipeline-dry domain=legal
```

## Pipeline Stages

The full pipeline runs four stages:

1. **Corpus Building** (`scripts/build_corpus.py`)
   - Load documents from configured sources
   - Quality filter (length, language)
   - Deduplicate with MinHash LSH
   - Format for training

2. **Synthetic Data Generation** (`scripts/generate_synthetic.py`)
   - Generate Q&A pairs from documents
   - Create instruction examples
   - Quality filter outputs

3. **Training** (`scripts/train_domain_model.py`)
   - Load base model with 4-bit quantization
   - Apply LoRA adapters
   - SFT training with domain data

4. **Evaluation** (`scripts/evaluate_domain.py`)
   - Run domain-specific benchmarks
   - Check knowledge retention
   - Measure terminology accuracy

## Individual Scripts

Run stages individually:

```bash
# Build corpus
python scripts/build_corpus.py --domain legal --output data/processed/legal_corpus

# Extract terminology
python scripts/extract_terminology.py \
    --corpus data/processed/legal_corpus \
    --output data/terminology/legal_terms.json

# Generate synthetic data
python scripts/generate_synthetic.py \
    --domain legal \
    --corpus data/processed/legal_corpus/corpus.jsonl \
    --output data/processed/legal_synthetic.jsonl

# Train model
python scripts/train_domain_model.py \
    --domain legal \
    --data data/processed/legal_synthetic.jsonl \
    --output models/legal-lora

# Evaluate
python scripts/evaluate_domain.py \
    --model models/legal-lora/final_adapter \
    --domain legal \
    --output reports/legal_eval.json
```

## Configuration

### Domain Configuration

Domain configs live in `configs/domains/`. Example (`configs/domains/legal.yaml`):

```yaml
name: legal
description: Legal domain adaptation

corpus:
  sources:
    - name: pile-of-law
      type: case_law
  quality_filters:
    min_length: 500
    max_length: 50000

terminology:
  min_frequency: 100

evaluation:
  benchmarks:
    - name: bar_exam
      passing_threshold: 0.7
```

### Training Configuration

Training configs in `configs/training/`. See `lora_adapt.yaml` and `instruction_tune.yaml`.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
make lint

# Run tests
make test

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration
```

## Project Structure

```
llm-fine-tuning-pipeline/
├── src/
│   ├── config/          # Pydantic config models
│   ├── corpus/          # Corpus building, terminology, synthetic data
│   ├── data/            # Loaders, formatters, tokenization
│   ├── models/          # Model loading utilities
│   ├── adaptation/      # LoRA, training strategies
│   └── evaluation/      # Benchmarks, retention, terminology eval
├── scripts/             # CLI entry points
├── configs/             # YAML configurations
│   ├── domains/         # Domain configs (legal.yaml, etc.)
│   └── training/        # Training configs
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
└── data/                # Data directories (raw, processed, terminology)
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA 12.1+ (for GPU training)
- 16GB+ VRAM recommended for training

## License

MIT
