"""Integration tests for the corpus building pipeline.

Tests the full chain: load → filter → deduplicate → format → save,
ensuring components work together correctly.
"""

import json
from pathlib import Path

from src.config.settings import QualityFilterConfig
from src.corpus.quality.deduplication import Deduplicator
from src.corpus.quality.filters import QualityFilter
from src.corpus.terminology.extractor import TerminologyExtractor
from src.data.formatters.domain_chat import DomainChatFormatter
from src.data.loaders.base_loader import Document, JSONLLoader


def _create_corpus_jsonl(path: Path, documents: list[dict]) -> None:
    """Write a corpus JSONL file for testing."""
    with open(path, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")


class TestCorpusBuildingPipeline:
    """Test the full corpus building pipeline end-to-end."""

    def test_load_filter_dedup_save(self, tmp_path: Path):
        """Full pipeline: load → filter → dedup → save."""
        # Create test corpus
        corpus_path = tmp_path / "corpus.jsonl"
        long_text = (
            "The court held that the defendant breached the contractual obligations "
            "by failing to deliver the goods on time as specified in the agreement. "
            "The plaintiff demonstrated clear and convincing evidence of damages."
        )
        documents = [
            {"text": long_text, "doc_id": "doc_1", "source": "test"},
            {"text": long_text, "doc_id": "doc_2", "source": "test"},  # duplicate
            {"text": "Short.", "doc_id": "doc_3", "source": "test"},  # too short
            {
                "text": (
                    "Quantum computing leverages superposition and entanglement to "
                    "perform computations that classical computers cannot. This represents "
                    "a paradigm shift in computational capability and algorithmic design."
                ),
                "doc_id": "doc_4",
                "source": "test",
            },
        ]
        _create_corpus_jsonl(corpus_path, documents)

        # Load
        loader = JSONLLoader(path=str(corpus_path), source_name="test")
        docs = loader.load()

        # Filter
        config = QualityFilterConfig(min_length=50, max_length=50000)
        quality_filter = QualityFilter(config, min_quality_score=0.0)
        docs = quality_filter.filter(docs)

        # Deduplicate
        deduplicator = Deduplicator(threshold=0.8, num_perm=128)
        docs = deduplicator.deduplicate(docs)

        # Save
        output_path = tmp_path / "output.jsonl"
        count = 0
        with open(output_path, "w") as f:
            for doc in docs:
                record = {"text": doc.text, "doc_id": doc.doc_id}
                f.write(json.dumps(record) + "\n")
                count += 1

        # Verify
        assert count == 2  # doc_1 kept, doc_2 removed (dup), doc_3 removed (short), doc_4 kept
        assert quality_filter.stats.dropped_too_short >= 1
        assert deduplicator.stats.removed >= 1

    def test_load_and_format(self, tmp_path: Path):
        """Pipeline: load → format to chat examples."""
        corpus_path = tmp_path / "corpus.jsonl"
        documents = [
            {
                "text": (
                    "The Supreme Court ruled on the interpretation of the Commerce Clause. "
                    "This landmark decision established new precedent for federal regulation."
                ),
                "doc_id": "doc_1",
                "source": "test",
                "metadata": {"title": "Commerce Clause Analysis", "category": "case_law"},
            },
        ]
        _create_corpus_jsonl(corpus_path, documents)

        # Load and format
        loader = JSONLLoader(path=str(corpus_path), source_name="test")
        formatter = DomainChatFormatter(domain="legal", mode="qa", max_length=8000)

        output_path = tmp_path / "formatted.jsonl"
        examples = formatter.format(loader.load())
        num = formatter.save_jsonl(examples, output_path)

        assert num >= 1

        # Verify format
        with open(output_path) as f:
            line = json.loads(f.readline())
        assert "messages" in line
        roles = [m["role"] for m in line["messages"]]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles


class TestTerminologyExtractionPipeline:
    """Test corpus → terminology extraction pipeline."""

    def test_extract_from_corpus(self, tmp_path: Path):
        """Load a corpus and extract terminology."""
        corpus_path = tmp_path / "corpus.jsonl"
        documents = [
            {
                "text": (
                    "The doctrine of respondeat superior holds employers liable. "
                    "Fiduciary duty requires loyalty and care. "
                    "Breach of contract occurs when obligations are not met."
                ),
                "doc_id": "doc_1",
                "source": "test",
            },
            {
                "text": (
                    "The plaintiff alleged breach of fiduciary duty. "
                    "Under the doctrine of respondeat superior, the employer is liable. "
                    "The court examined the contractual obligations."
                ),
                "doc_id": "doc_2",
                "source": "test",
            },
        ]
        _create_corpus_jsonl(corpus_path, documents)

        # Load
        loader = JSONLLoader(path=str(corpus_path), source_name="test")
        docs = loader.load_all()

        # Extract terms (min_frequency=1 for small test corpus)
        from src.config.settings import TerminologyConfig

        term_config = TerminologyConfig(min_frequency=1)
        extractor = TerminologyExtractor(config=term_config, use_spacy=False, max_ngram=3, top_k=50)
        terms = extractor.extract(docs)

        assert len(terms) > 0
        term_texts = [t.text for t in terms]
        # Should find some legal terms
        assert any("breach" in t or "doctrine" in t or "fiduciary" in t for t in term_texts)

        # Save and reload
        output_path = tmp_path / "terms.json"
        extractor.save(terms, str(output_path))

        with open(output_path) as f:
            data = json.load(f)
        assert data["num_terms"] == len(terms)


class TestFilterAndFormatPipeline:
    """Test filter → format → save pipeline."""

    def test_filter_then_format(self, tmp_path: Path):
        """Quality-filtered documents should format correctly."""
        good_text = (
            "The court examined the doctrine of promissory estoppel in this case. "
            "The defendant made a clear and definite promise to the plaintiff. "
            "The plaintiff reasonably relied on this promise to their detriment."
        )

        docs = [
            Document(text=good_text, doc_id="good", source="test", metadata={"title": "Estoppel"}),
            Document(text="Too short", doc_id="bad", source="test"),
        ]

        # Filter
        config = QualityFilterConfig(min_length=50, max_length=50000)
        qf = QualityFilter(config, min_quality_score=0.0)
        filtered = list(qf.filter(iter(docs)))
        assert len(filtered) == 1

        # Format
        formatter = DomainChatFormatter(domain="legal", mode="qa")
        examples = list(formatter.format(iter(filtered)))
        assert len(examples) >= 1

        # Verify the chat structure
        first = examples[0]
        assert len(first.messages) >= 2
        assert first.messages[-1].role == "assistant"
        assert "estoppel" in first.messages[-1].content.lower()
