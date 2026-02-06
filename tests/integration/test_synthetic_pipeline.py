"""Integration tests for the synthetic data generation pipeline.

Tests: corpus → Q&A generation → instruction generation → quality filtering → save.
"""

import json
from pathlib import Path

from src.corpus.synthetic.instruction_generator import InstructionGenerator, save_instructions
from src.corpus.synthetic.qa_generator import QAGenerator, save_qa_pairs
from src.corpus.synthetic.quality_filter import SyntheticQualityFilter
from src.data.loaders.base_loader import JSONLLoader


def _create_corpus_jsonl(path: Path, documents: list[dict]) -> None:
    with open(path, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")


SAMPLE_LEGAL_DOCS = [
    {
        "text": (
            "The Supreme Court addressed the question of whether federal preemption "
            "applies to state tort claims arising from pharmaceutical labeling. "
            "The majority opinion, authored by Justice Stevens, held that the federal "
            "regulatory framework does not bar state law claims for failure to warn. "
            "This decision established important precedent for pharmaceutical liability "
            "and the interplay between federal and state regulatory schemes."
        ),
        "doc_id": "case_001",
        "source": "courtlistener",
    },
    {
        "text": (
            "This Employment Agreement is entered into by and between the Company "
            "and the Employee. The Employee agrees to serve as Chief Technology Officer "
            "for a term of three years. Compensation shall include a base salary of "
            "two hundred thousand dollars per annum, plus equity grants as specified "
            "in Schedule A. Termination may occur for cause as defined in Section 5. "
            "The Employee shall be subject to non-compete restrictions for a period "
            "of twelve months following termination."
        ),
        "doc_id": "contract_001",
        "source": "sec_contracts",
    },
    {
        "text": (
            "The appellate court reversed the lower court's decision, finding that "
            "the trial court erred in granting summary judgment. The evidence presented "
            "by the plaintiff raised genuine issues of material fact regarding the "
            "defendant's negligence. The doctrine of res judicata did not apply because "
            "the prior case involved different parties and distinct factual circumstances."
        ),
        "doc_id": "case_002",
        "source": "courtlistener",
    },
]


class TestQAGenerationPipeline:
    """Test the full Q&A generation pipeline from corpus to filtered output."""

    def test_corpus_to_qa_pairs(self, tmp_path: Path):
        """Load corpus → generate Q&A → quality filter → save."""
        # Create corpus
        corpus_path = tmp_path / "corpus.jsonl"
        _create_corpus_jsonl(corpus_path, SAMPLE_LEGAL_DOCS)

        # Load
        loader = JSONLLoader(path=str(corpus_path), source_name="legal")
        docs = list(loader.load())
        assert len(docs) == 3

        # Generate Q&A pairs
        qa_gen = QAGenerator(domain="legal", max_pairs_per_doc=2, seed=42)
        all_pairs = []
        for doc in docs:
            pairs = qa_gen.generate_from_document(doc)
            all_pairs.extend(pairs)

        assert len(all_pairs) > 0

        # Quality filter
        qf = SyntheticQualityFilter(min_length=50, min_quality_score=0.3)
        filtered = qf.filter_qa_pairs(all_pairs, field="answer")
        assert len(filtered) > 0
        assert qf.stats.total > 0

        # Save
        output_path = tmp_path / "qa_pairs.jsonl"
        count = save_qa_pairs(iter(filtered), output_path, fmt="chat")
        assert count == len(filtered)

        # Verify output format
        with open(output_path) as f:
            for line in f:
                record = json.loads(line)
                assert "messages" in record
                messages = record["messages"]
                assert messages[-1]["role"] == "assistant"
                assert len(messages[-1]["content"]) > 0

    def test_streaming_generation(self, tmp_path: Path):
        """Test streaming generation with max_pairs limit."""
        corpus_path = tmp_path / "corpus.jsonl"
        _create_corpus_jsonl(corpus_path, SAMPLE_LEGAL_DOCS)

        loader = JSONLLoader(path=str(corpus_path), source_name="legal")
        qa_gen = QAGenerator(domain="legal", max_pairs_per_doc=5, seed=42)

        pairs = list(qa_gen.generate_from_documents(loader.load(), max_pairs=4))
        assert len(pairs) <= 4


class TestInstructionGenerationPipeline:
    """Test the full instruction generation pipeline."""

    def test_corpus_to_instructions(self, tmp_path: Path):
        """Load corpus → generate instructions → quality filter → save."""
        corpus_path = tmp_path / "corpus.jsonl"
        _create_corpus_jsonl(corpus_path, SAMPLE_LEGAL_DOCS)

        loader = JSONLLoader(path=str(corpus_path), source_name="legal")
        docs = list(loader.load())

        # Generate instruction pairs
        inst_gen = InstructionGenerator(domain="legal", max_examples_per_doc=2, seed=42)
        all_examples = []
        for doc in docs:
            examples = inst_gen.generate_from_document(doc)
            all_examples.extend(examples)

        assert len(all_examples) > 0
        # Verify task types are from the templates
        task_types = {ex.task_type for ex in all_examples}
        assert task_types.issubset(
            {"summarize", "analyze", "explain", "classify", "extract", "compare", "draft"}
        )

        # Quality filter
        qf = SyntheticQualityFilter(min_length=50, min_quality_score=0.3)
        filtered = qf.filter_qa_pairs(all_examples, field="response")
        assert len(filtered) > 0

        # Save
        output_path = tmp_path / "instructions.jsonl"
        count = save_instructions(iter(filtered), output_path, fmt="raw")
        assert count == len(filtered)

        # Verify output
        with open(output_path) as f:
            first = json.loads(f.readline())
        assert "instruction" in first
        assert "response" in first
        assert "task_type" in first

    def test_term_definitions_pipeline(self, tmp_path: Path):
        """Generate term definitions and save as chat format."""
        terms = ["respondeat superior", "res judicata", "fiduciary duty"]
        inst_gen = InstructionGenerator(
            domain="legal", terminology=terms, seed=42
        )

        definitions = inst_gen.generate_term_definitions()
        assert len(definitions) == 3

        # Save as chat format
        output_path = tmp_path / "term_defs.jsonl"
        count = save_instructions(
            iter(definitions), output_path,
            system_prompt="You are a legal assistant.",
            fmt="chat",
        )
        assert count == 3

        with open(output_path) as f:
            first = json.loads(f.readline())
        assert "messages" in first
        assert first["messages"][0]["role"] == "system"


class TestCombinedSyntheticPipeline:
    """Test combining QA and instruction generation with quality filtering."""

    def test_combined_output(self, tmp_path: Path):
        """Generate both QA and instruction pairs, merge into single output."""
        corpus_path = tmp_path / "corpus.jsonl"
        _create_corpus_jsonl(corpus_path, SAMPLE_LEGAL_DOCS)

        # Quality filter shared across both types
        qf = SyntheticQualityFilter(min_length=50, min_quality_score=0.3)

        # Generate Q&A pairs
        loader = JSONLLoader(path=str(corpus_path), source_name="legal")
        qa_gen = QAGenerator(domain="legal", max_pairs_per_doc=2, seed=42)
        qa_pairs = list(qa_gen.generate_from_documents(loader.load()))
        filtered_qa = qf.filter_qa_pairs(qa_pairs, field="answer")

        # Generate instruction pairs
        loader2 = JSONLLoader(path=str(corpus_path), source_name="legal")
        inst_gen = InstructionGenerator(domain="legal", max_examples_per_doc=2, seed=42)
        inst_examples = list(inst_gen.generate_from_documents(loader2.load()))
        filtered_inst = qf.filter_qa_pairs(inst_examples, field="response")

        # Merge
        output_path = tmp_path / "combined.jsonl"
        with open(output_path, "w") as f:
            for pair in filtered_qa:
                record = {"messages": pair.to_chat_messages()}
                f.write(json.dumps(record) + "\n")
            for ex in filtered_inst:
                record = {"messages": ex.to_chat_messages()}
                f.write(json.dumps(record) + "\n")

        # Verify
        with open(output_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        total = len(filtered_qa) + len(filtered_inst)
        assert len(lines) == total
        assert total > 0

        # All records should have valid chat format
        for record in lines:
            assert "messages" in record
            assert len(record["messages"]) >= 2
            assert record["messages"][-1]["role"] == "assistant"

    def test_quality_filter_stats_accumulate(self, tmp_path: Path):
        """Quality filter stats should accumulate across multiple filter calls."""
        corpus_path = tmp_path / "corpus.jsonl"
        _create_corpus_jsonl(corpus_path, SAMPLE_LEGAL_DOCS)

        qf = SyntheticQualityFilter(min_length=50, min_quality_score=0.3)

        # Filter QA pairs
        loader = JSONLLoader(path=str(corpus_path), source_name="legal")
        qa_gen = QAGenerator(domain="legal", seed=42)
        qa_pairs = list(qa_gen.generate_from_documents(loader.load()))
        qf.filter_qa_pairs(qa_pairs, field="answer")
        stats_after_qa = qf.stats.total

        # Filter instruction pairs (stats accumulate)
        loader2 = JSONLLoader(path=str(corpus_path), source_name="legal")
        inst_gen = InstructionGenerator(domain="legal", seed=42)
        inst_examples = list(inst_gen.generate_from_documents(loader2.load()))
        qf.filter_qa_pairs(inst_examples, field="response")

        assert qf.stats.total > stats_after_qa
