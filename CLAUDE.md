# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Domain Adaptation Toolkit** is a specialized LLM fine-tuning framework focused on adapting foundation models to specific professional domains. Unlike generic fine-tuning pipelines, this toolkit provides domain-specific data curation, terminology extraction, evaluation benchmarks, and guided fine-tuning strategies for legal, medical, financial, and scientific domains.

**Unique Focus: Domain-Specific LLM Adaptation**
- Automated domain corpus creation from professional sources
- Terminology extraction and embedding alignment
- Domain-specific evaluation benchmarks
- Multi-stage adaptation (continued pretraining → instruction tuning → RLHF)
- Knowledge retention testing to prevent catastrophic forgetting
- Compliance-aware training for regulated industries

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOMAIN CORPUS BUILDER                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Document │  │Terminology│  │ Quality  │  │ Synthetic│       │
│  │ Ingestion│  │ Extractor │  │  Filter  │  │Data Gen  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       └─────────────┴──────┬──────┴─────────────┘              │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-STAGE ADAPTATION                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Stage 1: Continued Pretraining                            │ │
│  │   Domain corpus → Causal LM → Domain-adapted base         │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Stage 2: Instruction Tuning                               │ │
│  │   Domain Q&A pairs → SFT → Domain instruction model       │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Stage 3: Alignment (Optional)                             │ │
│  │   Expert preferences → DPO/RLHF → Aligned domain model    │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOMAIN EVALUATION SUITE                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Domain   │  │ Knowledge│  │ Terminology│ │ Compliance│       │
│  │Benchmarks│  │ Retention│  │ Accuracy  │  │  Testing │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Domains

| Domain | Data Sources | Key Challenges | Evaluation |
|--------|--------------|----------------|------------|
| **Legal** | Case law, contracts, regulations | Precedent reasoning, jurisdiction | Bar exam, contract analysis |
| **Medical** | PubMed, clinical notes, guidelines | Patient safety, terminology | USMLE, diagnosis accuracy |
| **Financial** | SEC filings, earnings calls, reports | Numerical reasoning, compliance | CFA questions, sentiment |
| **Scientific** | ArXiv, journals, patents | Citation, technical accuracy | Domain QA, paper comprehension |
| **Code** | Documentation, issues, PRs | API usage, best practices | HumanEval, code review |

### Adaptation Strategies

| Strategy | When to Use | Memory | Quality |
|----------|-------------|--------|---------|
| Full Continued Pretraining | Large domain corpus available | Very High | Highest |
| LoRA Adaptation | Limited compute | Low | Good |
| Domain-Specific LoRA | Multiple domains needed | Low | Good |
| Mixture of Experts | Multi-domain with routing | High | Excellent |
| Vocabulary Extension | New terminology heavy | Medium | Very Good |

## Directory Structure

```
llm-fine-tuning-pipeline/
├── src/
│   ├── corpus/
│   │   ├── builders/
│   │   │   ├── legal_corpus.py          # Legal document processing
│   │   │   ├── medical_corpus.py        # PubMed, clinical notes
│   │   │   ├── financial_corpus.py      # SEC, earnings calls
│   │   │   └── scientific_corpus.py     # ArXiv, journals
│   │   ├── terminology/
│   │   │   ├── extractor.py             # Domain term extraction
│   │   │   ├── embeddings.py            # Term embedding alignment
│   │   │   └── glossary_builder.py      # Automated glossaries
│   │   ├── quality/
│   │   │   ├── filters.py               # Quality filtering
│   │   │   ├── deduplication.py         # Near-duplicate removal
│   │   │   └── toxicity_filter.py       # Content filtering
│   │   └── synthetic/
│   │       ├── qa_generator.py          # Q&A pair generation
│   │       ├── instruction_generator.py # Instruction data
│   │       └── preference_generator.py  # Preference pairs for DPO
│   ├── adaptation/
│   │   ├── stages/
│   │   │   ├── continued_pretrain.py    # Domain pretraining
│   │   │   ├── instruction_tune.py      # SFT stage
│   │   │   └── alignment.py             # DPO/RLHF stage
│   │   ├── strategies/
│   │   │   ├── full_finetune.py
│   │   │   ├── lora_adapt.py
│   │   │   ├── domain_lora.py           # Domain-specific adapters
│   │   │   └── vocab_extension.py       # Vocabulary expansion
│   │   └── curriculum/
│   │       ├── scheduler.py             # Curriculum learning
│   │       └── data_mixer.py            # Multi-source mixing
│   ├── evaluation/
│   │   ├── benchmarks/
│   │   │   ├── legal/
│   │   │   │   ├── bar_exam.py
│   │   │   │   ├── contract_analysis.py
│   │   │   │   └── case_reasoning.py
│   │   │   ├── medical/
│   │   │   │   ├── usmle.py
│   │   │   │   ├── diagnosis.py
│   │   │   │   └── drug_interaction.py
│   │   │   ├── financial/
│   │   │   │   ├── cfa_questions.py
│   │   │   │   ├── sentiment.py
│   │   │   │   └── numerical_reasoning.py
│   │   │   └── scientific/
│   │   │       ├── domain_qa.py
│   │   │       └── paper_comprehension.py
│   │   ├── retention/
│   │   │   ├── general_knowledge.py     # MMLU subset
│   │   │   ├── reasoning.py             # GSM8K, HellaSwag
│   │   │   └── catastrophic_forgetting.py
│   │   ├── terminology/
│   │   │   ├── term_accuracy.py         # Domain term usage
│   │   │   └── definition_quality.py
│   │   └── compliance/
│   │       ├── hallucination_rate.py
│   │       ├── citation_accuracy.py
│   │       └── disclaimer_generation.py
│   ├── data/
│   │   ├── loaders/
│   │   │   ├── pubmed_loader.py
│   │   │   ├── sec_loader.py
│   │   │   ├── arxiv_loader.py
│   │   │   └── legal_loader.py
│   │   ├── formatters/
│   │   │   ├── domain_chat.py           # Domain-aware chat format
│   │   │   └── citation_format.py       # Citation formatting
│   │   └── processors/
│   │       ├── tokenizer.py
│   │       └── domain_tokenizer.py      # Extended vocabulary
│   ├── models/
│   │   ├── loader.py
│   │   └── domain_configs/
│   │       ├── legal_llama.yaml
│   │       ├── med_llama.yaml
│   │       └── fin_mistral.yaml
│   ├── api/
│   │   ├── main.py
│   │   └── routes/
│   │       ├── train.py
│   │       ├── evaluate.py
│   │       └── corpus.py
│   └── config/
│       └── settings.py
├── configs/
│   ├── domains/
│   │   ├── legal.yaml
│   │   ├── medical.yaml
│   │   ├── financial.yaml
│   │   └── scientific.yaml
│   ├── training/
│   │   ├── continued_pretrain.yaml
│   │   ├── instruction_tune.yaml
│   │   └── dpo_align.yaml
│   └── evaluation/
│       └── domain_benchmarks.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── terminology/
│   └── benchmarks/
├── scripts/
│   ├── build_corpus.py
│   ├── extract_terminology.py
│   ├── train_domain_model.py
│   ├── evaluate_domain.py
│   └── generate_synthetic.py
├── notebooks/
│   ├── 01_corpus_analysis.ipynb
│   ├── 02_terminology_extraction.ipynb
│   ├── 03_training_analysis.ipynb
│   └── 04_domain_evaluation.ipynb
├── tests/
├── docker/
├── requirements.txt
└── README.md
```

## Commands

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install spaCy models for terminology extraction
python -m spacy download en_core_web_lg
```

### Corpus Building

```bash
# Build legal corpus from case law
python scripts/build_corpus.py \
    --domain legal \
    --sources courtlistener,sec_contracts \
    --output data/processed/legal_corpus \
    --min-quality 0.8

# Build medical corpus from PubMed
python scripts/build_corpus.py \
    --domain medical \
    --sources pubmed,guidelines \
    --output data/processed/medical_corpus \
    --filter-by-speciality cardiology,oncology

# Extract domain terminology
python scripts/extract_terminology.py \
    --corpus data/processed/legal_corpus \
    --output data/terminology/legal_terms.json \
    --min-frequency 100

# Generate synthetic instruction data
python scripts/generate_synthetic.py \
    --domain legal \
    --corpus data/processed/legal_corpus \
    --output data/processed/legal_instructions.jsonl \
    --num-samples 50000
```

### Multi-Stage Training

```bash
# Stage 1: Continued Pretraining on domain corpus
python scripts/train_domain_model.py \
    --stage pretrain \
    --config configs/training/continued_pretrain.yaml \
    --domain legal \
    --base-model meta-llama/Meta-Llama-3-8B \
    --corpus data/processed/legal_corpus \
    --output models/legal-llama-pretrained

# Stage 2: Instruction Tuning
python scripts/train_domain_model.py \
    --stage instruction \
    --config configs/training/instruction_tune.yaml \
    --domain legal \
    --base-model models/legal-llama-pretrained \
    --data data/processed/legal_instructions.jsonl \
    --output models/legal-llama-instruct

# Stage 3: DPO Alignment (optional)
python scripts/train_domain_model.py \
    --stage align \
    --config configs/training/dpo_align.yaml \
    --domain legal \
    --base-model models/legal-llama-instruct \
    --preferences data/processed/legal_preferences.jsonl \
    --output models/legal-llama-aligned

# Single-stage LoRA adaptation (faster)
python scripts/train_domain_model.py \
    --stage lora \
    --config configs/training/lora_adapt.yaml \
    --domain medical \
    --base-model meta-llama/Meta-Llama-3-8B-Instruct \
    --data data/processed/medical_instructions.jsonl \
    --output models/med-llama-lora
```

### Domain Evaluation

```bash
# Run domain-specific benchmarks
python scripts/evaluate_domain.py \
    --model models/legal-llama-aligned \
    --domain legal \
    --benchmarks bar_exam,contract_analysis,case_reasoning \
    --output reports/legal_eval.json

# Check knowledge retention (catastrophic forgetting)
python scripts/evaluate_domain.py \
    --model models/legal-llama-aligned \
    --benchmarks mmlu,hellaswag,gsm8k \
    --output reports/retention_eval.json

# Terminology accuracy evaluation
python scripts/evaluate_domain.py \
    --model models/legal-llama-aligned \
    --eval-type terminology \
    --glossary data/terminology/legal_terms.json \
    --output reports/terminology_eval.json

# Compliance testing (hallucination rate)
python scripts/evaluate_domain.py \
    --model models/medical-llama \
    --eval-type compliance \
    --domain medical \
    --output reports/compliance_eval.json
```

## Configuration

### Domain Configuration

```yaml
# configs/domains/legal.yaml
domain:
  name: legal
  description: Legal domain adaptation for contract analysis and case law

corpus:
  sources:
    - name: courtlistener
      type: case_law
      jurisdictions: [federal, state]
      date_range: [2010-01-01, 2024-01-01]
    - name: sec_contracts
      type: contracts
      categories: [merger, employment, licensing]

  quality_filters:
    min_length: 500
    max_length: 50000
    language: en
    dedup_threshold: 0.85

terminology:
  extraction:
    method: tfidf_ner_hybrid
    min_frequency: 100
    pos_tags: [NOUN, PROPN]
  glossary_sources:
    - blacks_law_dictionary
    - legal_information_institute

evaluation:
  benchmarks:
    - name: bar_exam
      weight: 0.3
      passing_threshold: 0.7
    - name: contract_analysis
      weight: 0.4
      passing_threshold: 0.8
    - name: case_reasoning
      weight: 0.3
      passing_threshold: 0.75

  retention:
    - mmlu_professional_law
    - mmlu_general
    - hellaswag

compliance:
  require_citations: true
  disclaimer_required: true
  hallucination_threshold: 0.05
```

### Training Configuration

```yaml
# configs/training/continued_pretrain.yaml
training:
  stage: continued_pretraining

  data:
    domain_weight: 0.8
    general_weight: 0.2  # Prevent forgetting
    max_length: 4096
    packing: true

  model:
    torch_dtype: bfloat16
    attn_implementation: flash_attention_2
    gradient_checkpointing: true

  optimizer:
    name: adamw
    learning_rate: 1e-5  # Lower for continued pretraining
    weight_decay: 0.1
    warmup_ratio: 0.05
    lr_scheduler: cosine

  training:
    num_epochs: 1
    batch_size: 4
    gradient_accumulation_steps: 8
    max_steps: 50000

  curriculum:
    enabled: true
    strategy: difficulty_ascending
    stages:
      - name: basic_documents
        steps: 10000
        filter: length < 2000
      - name: complex_documents
        steps: 40000
        filter: length >= 2000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/corpus/build` | POST | Build domain corpus |
| `/api/v1/corpus/status/{job_id}` | GET | Get corpus build status |
| `/api/v1/terminology/extract` | POST | Extract domain terminology |
| `/api/v1/train` | POST | Start domain adaptation training |
| `/api/v1/train/{job_id}` | GET | Get training status |
| `/api/v1/evaluate` | POST | Run domain evaluation |
| `/api/v1/evaluate/retention` | POST | Run retention evaluation |
| `/api/v1/models` | GET | List domain-adapted models |
| `/api/v1/models/{name}/benchmark` | GET | Get model benchmark results |

## Domain Evaluation Metrics

| Domain | Primary Metrics | Target |
|--------|----------------|--------|
| Legal | Bar exam pass rate, Contract F1 | >70%, >85% |
| Medical | USMLE score, Diagnosis accuracy | >60%, >80% |
| Financial | CFA questions, Sentiment F1 | >65%, >90% |
| Scientific | Domain QA accuracy | >75% |
| All | Knowledge retention (MMLU delta) | <5% drop |
| All | Hallucination rate | <5% |

## Implementation Phases

### Phase 1: Corpus Infrastructure
- [ ] Multi-source document loaders
- [ ] Quality filtering pipeline
- [ ] Deduplication system
- [ ] Terminology extraction

### Phase 2: Training Pipeline
- [ ] Continued pretraining stage
- [ ] LoRA domain adaptation
- [ ] Curriculum learning
- [ ] Data mixing strategies

### Phase 3: Instruction Tuning
- [ ] Synthetic Q&A generation
- [ ] Domain-specific SFT
- [ ] Multi-turn conversation data
- [ ] Citation-aware training

### Phase 4: Evaluation Suite
- [ ] Domain-specific benchmarks
- [ ] Knowledge retention tests
- [ ] Terminology accuracy
- [ ] Compliance testing

### Phase 5: Advanced Adaptation
- [ ] DPO/RLHF alignment
- [ ] Vocabulary extension
- [ ] Multi-domain routing
- [ ] Production deployment

## Dependencies

```
# Core
torch>=2.2.0
transformers>=4.38.0
peft>=0.8.0
trl>=0.7.0
accelerate>=0.27.0
bitsandbytes>=0.42.0

# Data Processing
datasets>=2.17.0
spacy>=3.7.0
scispacy>=0.5.0
pubmed-parser>=0.3.0

# Evaluation
lm-eval>=0.4.0
rouge-score>=0.1.2

# Domain-Specific
legal-bert>=0.1.0
sec-api>=1.0.0

# Training
deepspeed>=0.13.0
flash-attn>=2.5.0
wandb>=0.16.0
```

## Hardware Requirements

| Stage | Model Size | Min VRAM | Recommended |
|-------|------------|----------|-------------|
| Continued Pretrain (Full) | 7B | 80GB | 8x A100 |
| Continued Pretrain (LoRA) | 7B | 24GB | 1x A100 |
| Instruction Tune (LoRA) | 7B | 16GB | 1x A100 |
| DPO Alignment | 7B | 24GB | 1x A100 |
| Evaluation | 7B | 16GB | 1x A100 |

## Testing

```bash
# Unit tests
pytest tests/unit -v

# Integration tests (requires GPU)
pytest tests/integration -v --gpu

# Test corpus building
pytest tests/unit/test_corpus.py -v

# Test evaluation benchmarks
pytest tests/unit/test_evaluation.py -v

# End-to-end domain adaptation test
pytest tests/e2e/test_adaptation.py -v --domain legal --small-model
```
