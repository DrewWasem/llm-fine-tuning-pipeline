"""Tests for quality filtering and deduplication."""


from src.config.settings import QualityFilterConfig
from src.corpus.quality.deduplication import Deduplicator, MinHashSignature, _ngrams
from src.corpus.quality.filters import (
    QualityFilter,
    _detect_language_heuristic,
    _quality_score,
)
from src.data.loaders.base_loader import Document

# ── Quality score tests ────────────────────────────────────────────────────────


class TestQualityScore:
    def test_normal_text_high_score(self):
        text = (
            "The court held that the defendant's actions constituted a breach "
            "of the contractual obligations as outlined in Section 4.2 of the "
            "agreement. The plaintiff demonstrated clear damages resulting from "
            "this breach, and the evidence presented was sufficient to establish "
            "liability under the applicable state law."
        )
        score = _quality_score(text)
        assert score >= 0.5

    def test_empty_text_zero_score(self):
        assert _quality_score("") == 0.0

    def test_repetitive_text_penalized(self):
        text = ("duplicate line\n" * 50)
        score = _quality_score(text)
        assert score < 0.7

    def test_special_chars_penalized(self):
        text = "!!@@##$$%%^^&&**" * 100
        score = _quality_score(text)
        assert score < 1.0  # penalized for special chars

    def test_score_range(self):
        texts = ["hello world", "a" * 100, "The quick brown fox jumps over the lazy dog."]
        for text in texts:
            score = _quality_score(text)
            assert 0.0 <= score <= 1.0


# ── Language detection tests ───────────────────────────────────────────────────


class TestLanguageDetection:
    def test_english_text(self):
        text = "The court held that the defendant breached the contract and is liable for damages."
        assert _detect_language_heuristic(text) == "en"

    def test_empty_text(self):
        assert _detect_language_heuristic("") == "unknown"

    def test_numbers_only(self):
        assert _detect_language_heuristic("12345 67890") == "unknown"


# ── QualityFilter integration tests ───────────────────────────────────────────


def _make_doc(text: str, doc_id: str = "test") -> Document:
    return Document(text=text, doc_id=doc_id, source="test")


class TestQualityFilter:
    def test_filter_too_short(self):
        config = QualityFilterConfig(min_length=100, max_length=50000)
        qf = QualityFilter(config)
        docs = [_make_doc("short")]
        result = list(qf.filter(iter(docs)))
        assert len(result) == 0
        assert qf.stats.dropped_too_short == 1

    def test_filter_too_long(self):
        config = QualityFilterConfig(min_length=1, max_length=100)
        qf = QualityFilter(config)
        docs = [_make_doc("x" * 200)]
        result = list(qf.filter(iter(docs)))
        assert len(result) == 0
        assert qf.stats.dropped_too_long == 1

    def test_pass_good_documents(self):
        config = QualityFilterConfig(min_length=10, max_length=50000)
        qf = QualityFilter(config, min_quality_score=0.0)
        text = "The court held that the defendant is liable for breach of contract under state law. " * 10
        docs = [_make_doc(text)]
        result = list(qf.filter(iter(docs)))
        assert len(result) == 1
        assert qf.stats.passed == 1

    def test_filter_stats(self):
        config = QualityFilterConfig(min_length=50, max_length=50000)
        qf = QualityFilter(config, min_quality_score=0.0)
        docs = [
            _make_doc("too short"),
            _make_doc("This is a sufficiently long document for our filters to accept. " * 3),
            _make_doc("Another good document with enough content to pass the length filter. " * 3),
        ]
        list(qf.filter(iter(docs)))  # consume the iterator
        assert qf.stats.total_input == 3
        assert qf.stats.passed == 2
        assert qf.stats.dropped_too_short == 1


# ── MinHash / n-gram tests ─────────────────────────────────────────────────────


class TestNGrams:
    def test_basic_ngrams(self):
        result = _ngrams("hello world", n=5)
        assert "hello" in result
        assert " worl" in result

    def test_short_text(self):
        result = _ngrams("hi", n=5)
        assert result == {"hi"}

    def test_empty_text(self):
        result = _ngrams("", n=5)
        assert result == {""}


class TestMinHashSignature:
    def test_identical_texts_same_signature(self):
        mh = MinHashSignature(num_perm=64)
        sig1 = mh.compute("the quick brown fox")
        sig2 = mh.compute("the quick brown fox")
        assert sig1 == sig2

    def test_similar_texts_high_similarity(self):
        mh = MinHashSignature(num_perm=128)
        sig1 = mh.compute("the quick brown fox jumps over the lazy dog")
        sig2 = mh.compute("the quick brown fox jumps over the lazy cat")
        similarity = MinHashSignature.jaccard_estimate(sig1, sig2)
        assert similarity > 0.5

    def test_different_texts_low_similarity(self):
        mh = MinHashSignature(num_perm=128)
        sig1 = mh.compute("the quick brown fox jumps over the lazy dog in the park")
        sig2 = mh.compute("quantum mechanics describes the behavior of particles at atomic scales")
        similarity = MinHashSignature.jaccard_estimate(sig1, sig2)
        assert similarity < 0.5


# ── Deduplicator tests ────────────────────────────────────────────────────────


class TestDeduplicator:
    def test_remove_exact_duplicates(self):
        dedup = Deduplicator(threshold=0.8, num_perm=128)
        text = "This is a legal document about contract law and its applications in modern jurisprudence."
        docs = [
            _make_doc(text, doc_id="a"),
            _make_doc(text, doc_id="b"),  # exact duplicate
        ]
        result = list(dedup.deduplicate(iter(docs)))
        assert len(result) == 1
        assert result[0].doc_id == "a"
        assert dedup.stats.removed == 1

    def test_keep_different_documents(self):
        dedup = Deduplicator(threshold=0.8, num_perm=128)
        docs = [
            _make_doc(
                "The plaintiff filed a motion for summary judgment arguing that the defendant "
                "failed to perform contractual obligations under the merger agreement.",
                doc_id="a",
            ),
            _make_doc(
                "Quantum computing leverages quantum mechanical phenomena such as superposition "
                "and entanglement to process information in fundamentally different ways.",
                doc_id="b",
            ),
        ]
        result = list(dedup.deduplicate(iter(docs)))
        assert len(result) == 2

    def test_dedup_stats(self):
        dedup = Deduplicator(threshold=0.8, num_perm=128)
        text = "Repeated legal document about breach of fiduciary duty in corporate governance matters."
        docs = [_make_doc(text, doc_id=str(i)) for i in range(5)]
        list(dedup.deduplicate(iter(docs)))  # consume the iterator
        assert dedup.stats.total_input == 5
        assert dedup.stats.kept == 1
        assert dedup.stats.removed == 4

    def test_empty_stream(self):
        dedup = Deduplicator()
        result = list(dedup.deduplicate(iter([])))
        assert len(result) == 0
