"""Tests for terminology extraction."""

from src.config.settings import TerminologyConfig
from src.corpus.terminology.extractor import (
    TerminologyExtractor,
    _extract_ngrams,
    _tokenize_simple,
)
from src.data.loaders.base_loader import Document

# ── Tokenizer tests ───────────────────────────────────────────────────────────


class TestTokenizer:
    def test_basic_tokenization(self):
        tokens = _tokenize_simple("The court held that breach-of-contract is actionable.")
        assert "the" in tokens
        assert "court" in tokens
        assert "breach-of-contract" in tokens

    def test_empty_text(self):
        assert _tokenize_simple("") == []


class TestNGrams:
    def test_unigrams(self):
        tokens = ["the", "court", "held"]
        result = _extract_ngrams(tokens, min_n=1, max_n=1)
        assert result == ["the", "court", "held"]

    def test_bigrams(self):
        tokens = ["the", "court", "held"]
        result = _extract_ngrams(tokens, min_n=2, max_n=2)
        assert "the court" in result
        assert "court held" in result
        assert len(result) == 2

    def test_trigrams(self):
        tokens = ["the", "court", "held"]
        result = _extract_ngrams(tokens, min_n=3, max_n=3)
        assert result == ["the court held"]


# ── TerminologyExtractor tests ────────────────────────────────────────────────


def _legal_docs() -> list[Document]:
    """Create sample legal documents for testing."""
    return [
        Document(
            text=(
                "The court held that the defendant breached the fiduciary duty owed to "
                "the plaintiff. The breach of fiduciary duty was established through "
                "clear and convincing evidence. The court awarded damages for the breach "
                "of fiduciary duty. The fiduciary duty standard requires loyalty and care."
            ),
            doc_id="doc_0",
        ),
        Document(
            text=(
                "Summary judgment was granted in favor of the plaintiff based on the "
                "undisputed material facts. The court applied the summary judgment "
                "standard as outlined in the Federal Rules of Civil Procedure. The "
                "summary judgment motion demonstrated no genuine dispute of material fact."
            ),
            doc_id="doc_1",
        ),
        Document(
            text=(
                "The arbitration clause in the contract was found to be enforceable. "
                "The court compelled arbitration pursuant to the Federal Arbitration Act. "
                "The arbitration clause covered all disputes arising from the contract. "
                "Compulsory arbitration was ordered for the breach of contract claim."
            ),
            doc_id="doc_2",
        ),
    ]


class TestTerminologyExtractor:
    def test_extract_basic_terms(self):
        config = TerminologyConfig(min_frequency=2)
        extractor = TerminologyExtractor(config=config, use_spacy=False, top_k=100)
        terms = extractor.extract(_legal_docs())
        assert len(terms) > 0

    def test_terms_have_tfidf_scores(self):
        config = TerminologyConfig(min_frequency=2)
        extractor = TerminologyExtractor(config=config, use_spacy=False)
        terms = extractor.extract(_legal_docs())
        for term in terms:
            assert term.tfidf_score > 0

    def test_terms_sorted_by_score(self):
        config = TerminologyConfig(min_frequency=2)
        extractor = TerminologyExtractor(config=config, use_spacy=False)
        terms = extractor.extract(_legal_docs())
        scores = [t.tfidf_score for t in terms]
        assert scores == sorted(scores, reverse=True)

    def test_min_frequency_filter(self):
        config = TerminologyConfig(min_frequency=100)
        extractor = TerminologyExtractor(config=config, use_spacy=False)
        terms = extractor.extract(_legal_docs())
        # With min_frequency=100, no terms should survive from 3 short docs
        assert len(terms) == 0

    def test_top_k_limit(self):
        config = TerminologyConfig(min_frequency=2)
        extractor = TerminologyExtractor(config=config, use_spacy=False, top_k=5)
        terms = extractor.extract(_legal_docs())
        assert len(terms) <= 5

    def test_empty_documents(self):
        config = TerminologyConfig(min_frequency=1)
        extractor = TerminologyExtractor(config=config, use_spacy=False)
        terms = extractor.extract([])
        assert len(terms) == 0

    def test_save_and_load(self, tmp_path):
        config = TerminologyConfig(min_frequency=2)
        extractor = TerminologyExtractor(config=config, use_spacy=False, top_k=10)
        terms = extractor.extract(_legal_docs())

        output_path = tmp_path / "terms.json"
        extractor.save(terms, output_path)

        assert output_path.exists()
        import json

        with open(output_path) as f:
            data = json.load(f)
        assert data["num_terms"] == len(terms)
        assert len(data["terms"]) == len(terms)

    def test_term_to_dict(self):
        config = TerminologyConfig(min_frequency=2)
        extractor = TerminologyExtractor(config=config, use_spacy=False)
        terms = extractor.extract(_legal_docs())
        if terms:
            d = terms[0].to_dict()
            assert "text" in d
            assert "frequency" in d
            assert "tfidf_score" in d
