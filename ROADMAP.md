# MVP Roadmap — Domain Adaptation Toolkit

## MVP Goal

Get a **single domain (Legal) working end-to-end**: load documents, build a corpus, fine-tune a model with LoRA, and evaluate it against a domain benchmark. One domain, one training strategy, one evaluation — shipped.

---

## Milestone 0: Project Scaffolding ✅
**Goal**: Bootable project with CI, deps, and structure.

- [x] Create `requirements.txt` with pinned core deps (torch, transformers, peft, trl, accelerate, datasets, spacy)
- [x] Create `pyproject.toml` or `setup.py` for package installation
- [x] Set up directory structure with `__init__.py` files
- [x] Create `src/config/settings.py` with Pydantic-based config loading
- [x] Create `configs/domains/legal.yaml` (first domain config)
- [x] Create `configs/training/lora_adapt.yaml` (simplest training config)
- [x] Add basic `Dockerfile` for reproducible environment
- [x] Add `Makefile` with common commands (`make install`, `make test`, `make lint`)
- [x] Set up GitHub repo, push initial commit
- [x] Add pre-commit hooks (ruff, mypy)

**Deliverable**: `pip install -e .` works, `pytest` runs (0 tests, 0 errors), configs load.

---

## Milestone 1: Data Loading & Corpus Building ✅
**Goal**: Ingest raw legal documents and produce a clean, deduplicated corpus.

### 1a — Data Loader
- [x] Implement `src/data/loaders/base_loader.py` — abstract interface for all loaders
- [x] Implement `src/data/loaders/legal_loader.py` — load legal docs from local files / HuggingFace datasets (e.g., `pile-of-law`)
- [x] Add CLI entry point: `python scripts/build_corpus.py --domain legal --source pile-of-law`

### 1b — Quality Filtering
- [x] Implement `src/corpus/quality/filters.py` — length filter, language filter, quality score
- [x] Implement `src/corpus/quality/deduplication.py` — MinHash-based near-duplicate removal

### 1c — Terminology Extraction
- [x] Implement `src/corpus/terminology/extractor.py` — TF-IDF + spaCy NER hybrid extraction
- [x] Add CLI: `python scripts/extract_terminology.py --corpus data/processed/legal_corpus`
- [x] Output: `data/terminology/legal_terms.json`

### 1d — Data Formatting
- [x] Implement `src/data/formatters/domain_chat.py` — convert raw docs to training format (chat template)
- [x] Implement `src/data/processors/tokenizer.py` — tokenization with HF tokenizer, packing support

**Deliverable**: Run `build_corpus.py` → get a clean `.jsonl` corpus + terminology file. Unit tests for filters and dedup.

---

## Milestone 2: Training Pipeline (LoRA) ✅
**Goal**: Fine-tune a base model on the legal corpus using LoRA. One strategy, one stage.

### 2a — Model Loading
- [x] Implement `src/models/loader.py` — load base model + tokenizer with quantization support (4-bit via bitsandbytes)
- [x] Create `src/models/domain_configs/legal_llama.yaml` — model-specific params

### 2b — LoRA Training
- [x] Implement `src/adaptation/strategies/lora_adapt.py` — LoRA fine-tuning using `peft` + `trl`
- [x] Implement `src/adaptation/stages/instruction_tune.py` — SFT training loop with `SFTTrainer`
- [x] Add training config: `configs/training/instruction_tune.yaml`
- [x] Integrate W&B logging for loss curves, learning rate, GPU utilization

### 2c — Data Mixing
- [x] Implement `src/adaptation/curriculum/data_mixer.py` — mix domain data with general data (configurable ratio) to prevent catastrophic forgetting

### 2d — Training Script
- [x] Implement `scripts/train_domain_model.py` — CLI orchestrator
  - Loads config → loads model → loads data → trains → saves adapter + merged model
- [x] Support checkpoint resumption

**Deliverable**: Run `train_domain_model.py` → get a LoRA adapter saved to disk. Training logs in W&B.

---

## Milestone 3: Evaluation ✅
**Goal**: Measure whether the adapted model actually improved on legal tasks, and didn't forget general knowledge.

### 3a — Domain Benchmark
- [x] Implement `src/evaluation/benchmarks/legal/bar_exam.py` — multiple-choice legal QA (can use MMLU professional_law subset)
- [x] Implement `src/evaluation/benchmarks/legal/contract_analysis.py` — contract clause extraction/classification

### 3b — Knowledge Retention
- [x] Implement `src/evaluation/retention/general_knowledge.py` — MMLU subset to check general knowledge retention
- [x] Implement `src/evaluation/retention/catastrophic_forgetting.py` — compare base vs adapted model on general benchmarks

### 3c — Terminology Accuracy
- [x] Implement `src/evaluation/terminology/term_accuracy.py` — test if model uses domain terms correctly in generated text

### 3d — Evaluation Script
- [x] Implement `scripts/evaluate_domain.py` — CLI orchestrator
  - Loads model → runs benchmarks → outputs JSON report
- [x] Output a comparison: base model vs adapted model, per-benchmark scores

**Deliverable**: Run `evaluate_domain.py` → get a JSON report showing domain score improvement and retention delta.

---

## Milestone 4: Synthetic Data Generation
**Goal**: Generate domain-specific instruction data to improve instruction-following in the domain.

- [x] Implement `src/corpus/synthetic/qa_generator.py` — generate Q&A pairs from legal documents using an LLM (OpenAI API or local model)
- [x] Implement `src/corpus/synthetic/instruction_generator.py` — generate instruction-response pairs for SFT
- [x] Add CLI: `python scripts/generate_synthetic.py --domain legal --num-samples 10000`
- [x] Add quality filtering for generated data (reject low-quality generations)

**Deliverable**: Run `generate_synthetic.py` → get `legal_instructions.jsonl` ready for training.

---

## Milestone 5: End-to-End Integration & Tests ✅
**Goal**: The full pipeline runs with a single command. Tests cover critical paths.

### 5a — Integration
- [x] Create `scripts/run_pipeline.py` — orchestrates corpus → train → evaluate in sequence
- [x] Add `configs/domains/legal.yaml` as a complete pipeline config (all stages)

### 5b — Tests
- [x] Unit tests for corpus building (filters, dedup, terminology)
- [x] Unit tests for data formatting and tokenization
- [x] Unit tests for evaluation metric calculations
- [x] Integration test: mini pipeline with a tiny model (e.g., `TinyLlama`) on a small dataset
- [x] Add CI workflow (GitHub Actions) — lint + unit tests on every PR

### 5c — Documentation
- [x] Write `README.md` — quick start, installation, usage examples
- [x] Add example output / sample evaluation report

**Deliverable**: `make pipeline domain=legal` runs the full thing. CI is green. README has a quick start.

---

## What's NOT in the MVP

These are deferred to post-MVP iterations:

| Deferred | Why |
|----------|-----|
| Medical, Financial, Scientific, Code domains | Prove it works on one domain first |
| Continued pretraining stage | LoRA instruction tuning is faster to validate |
| DPO/RLHF alignment stage | Needs preference data, more complex |
| Full fine-tuning strategy | Requires multi-GPU, expensive |
| Vocabulary extension | Optimization, not core pipeline |
| Mixture of Experts routing | Multi-domain feature, not needed for 1 domain |
| REST API (`src/api/`) | CLI-first for MVP, API is a wrapper |
| Compliance testing (hallucination, citations) | Important but secondary to core pipeline |
| Toxicity filtering | Nice-to-have for data quality |
| Notebooks | Documentation, not core functionality |
| Docker Compose / multi-service setup | Single-machine first |

---

## Suggested Sprint Breakdown

| Sprint | Milestone | Duration | Output |
|--------|-----------|----------|--------|
| **Sprint 1** | M0: Scaffolding | 3-4 days | Bootable project |
| **Sprint 2** | M1: Data & Corpus | 1-2 weeks | Clean legal corpus |
| **Sprint 3** | M2: Training | 1-2 weeks | Trained LoRA adapter |
| **Sprint 4** | M3: Evaluation | 1 week | Benchmark results |
| **Sprint 5** | M4: Synthetic Data | 1 week | Instruction dataset |
| **Sprint 6** | M5: Integration | 1 week | E2E pipeline + CI |

**Total estimated: ~6-8 weeks to MVP**

---

## Success Criteria

The MVP is done when:

1. **Pipeline runs end-to-end**: corpus build → LoRA train → evaluate, for the legal domain
2. **Measurable improvement**: adapted model scores higher on legal benchmarks than base model
3. **No catastrophic forgetting**: general knowledge retention stays within 5% of base model
4. **Reproducible**: another developer can clone, install, and run the pipeline following the README
5. **Tested**: CI passes with unit + integration tests covering critical paths
