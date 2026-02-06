"""Deep tests for deduplication: LSHIndex, DedupStats, hash_shingle, edge cases."""

from pytest import approx

from src.corpus.quality.deduplication import (
    Deduplicator,
    DedupStats,
    LSHIndex,
    MinHashSignature,
    _hash_shingle,
    _ngrams,
)
from src.data.loaders.base_loader import Document


def _make_doc(text: str, doc_id: str = "test") -> Document:
    return Document(text=text, doc_id=doc_id, source="test")


# ── _hash_shingle tests ─────────────────────────────────────────────────────


class TestHashShingle:
    def test_deterministic(self):
        assert _hash_shingle("hello") == _hash_shingle("hello")

    def test_different_inputs(self):
        assert _hash_shingle("hello") != _hash_shingle("world")

    def test_returns_int(self):
        result = _hash_shingle("test shingle")
        assert isinstance(result, int)
        assert result >= 0

    def test_32_bit_range(self):
        result = _hash_shingle("any text here")
        assert 0 <= result <= (1 << 32) - 1


# ── _ngrams edge cases ──────────────────────────────────────────────────────


class TestNGramsEdgeCases:
    def test_text_exactly_n_length(self):
        result = _ngrams("abcde", n=5)
        assert result == {"abcde"}

    def test_text_one_longer_than_n(self):
        result = _ngrams("abcdef", n=5)
        assert len(result) == 2
        assert "abcde" in result
        assert "bcdef" in result

    def test_whitespace_only(self):
        result = _ngrams("     ", n=5)
        # After strip + lower: ""
        assert result == {""}

    def test_case_normalization(self):
        result = _ngrams("HELLO WORLD", n=5)
        # Should be lowercase
        assert all(s == s.lower() for s in result)

    def test_large_n(self):
        result = _ngrams("short", n=100)
        assert result == {"short"}


# ── LSHIndex tests ───────────────────────────────────────────────────────────


class TestLSHIndex:
    def test_insert_first_doc_no_candidates(self):
        lsh = LSHIndex(num_bands=4, num_rows=2)
        sig = [1, 2, 3, 4, 5, 6, 7, 8]
        candidates = lsh.insert("doc_0", sig)
        assert candidates == set()

    def test_insert_identical_sigs_returns_candidate(self):
        lsh = LSHIndex(num_bands=4, num_rows=2)
        sig = [1, 2, 3, 4, 5, 6, 7, 8]
        lsh.insert("doc_0", sig)
        candidates = lsh.insert("doc_1", sig)
        assert "doc_0" in candidates

    def test_insert_different_sigs_no_candidate(self):
        lsh = LSHIndex(num_bands=4, num_rows=2)
        sig1 = [1, 2, 3, 4, 5, 6, 7, 8]
        sig2 = [100, 200, 300, 400, 500, 600, 700, 800]
        lsh.insert("doc_0", sig1)
        candidates = lsh.insert("doc_1", sig2)
        assert "doc_0" not in candidates

    def test_insert_partially_matching_bands(self):
        lsh = LSHIndex(num_bands=4, num_rows=2)
        sig1 = [1, 2, 3, 4, 5, 6, 7, 8]
        # Same first band, different rest
        sig2 = [1, 2, 99, 99, 99, 99, 99, 99]
        lsh.insert("doc_0", sig1)
        candidates = lsh.insert("doc_1", sig2)
        # First band matches (1,2), so doc_0 should be a candidate
        assert "doc_0" in candidates

    def test_multiple_docs_in_same_bucket(self):
        lsh = LSHIndex(num_bands=4, num_rows=2)
        sig = [10, 20, 30, 40, 50, 60, 70, 80]
        lsh.insert("doc_a", sig)
        lsh.insert("doc_b", sig)
        candidates = lsh.insert("doc_c", sig)
        assert "doc_a" in candidates
        assert "doc_b" in candidates

    def test_num_bands_and_rows(self):
        lsh = LSHIndex(num_bands=8, num_rows=4)
        assert lsh.num_bands == 8
        assert lsh.num_rows == 4
        assert len(lsh._buckets) == 8


# ── DedupStats tests ─────────────────────────────────────────────────────────


class TestDedupStats:
    def test_dedup_rate(self):
        stats = DedupStats(total_input=100, kept=80, removed=20)
        assert stats.dedup_rate == approx(0.2)

    def test_dedup_rate_zero_input(self):
        stats = DedupStats()
        assert stats.dedup_rate == 0.0

    def test_dedup_rate_no_removals(self):
        stats = DedupStats(total_input=50, kept=50, removed=0)
        assert stats.dedup_rate == 0.0

    def test_summary(self):
        stats = DedupStats(total_input=10, kept=7, removed=3)
        s = stats.summary()
        assert s["total_input"] == 10
        assert s["kept"] == 7
        assert s["removed"] == 3
        assert s["dedup_rate"] == approx(0.3)


# ── MinHashSignature edge cases ──────────────────────────────────────────────


class TestMinHashSignatureEdgeCases:
    def test_empty_text_signature(self):
        mh = MinHashSignature(num_perm=16)
        sig = mh.compute("")
        assert len(sig) == 16

    def test_single_character(self):
        mh = MinHashSignature(num_perm=16)
        sig = mh.compute("x")
        assert len(sig) == 16

    def test_signature_length_matches_num_perm(self):
        for num_perm in [8, 32, 64, 128]:
            mh = MinHashSignature(num_perm=num_perm)
            sig = mh.compute("test document text")
            assert len(sig) == num_perm

    def test_jaccard_identical_signatures(self):
        sig = [1, 2, 3, 4, 5]
        assert MinHashSignature.jaccard_estimate(sig, sig) == 1.0

    def test_jaccard_completely_different(self):
        sig1 = [1, 2, 3, 4, 5]
        sig2 = [6, 7, 8, 9, 10]
        assert MinHashSignature.jaccard_estimate(sig1, sig2) == 0.0

    def test_jaccard_mismatched_lengths_raises(self):
        import pytest

        with pytest.raises(ValueError):
            MinHashSignature.jaccard_estimate([1, 2], [1, 2, 3])


# ── Deduplicator edge cases ─────────────────────────────────────────────────


class TestDeduplicatorEdgeCases:
    def test_threshold_zero_keeps_all(self):
        """With threshold=1.0, only exact MinHash matches are removed."""
        dedup = Deduplicator(threshold=1.0, num_perm=128)
        docs = [
            _make_doc("Document about contract law principles.", doc_id="a"),
            _make_doc("Document about contract law principles and more.", doc_id="b"),
        ]
        result = list(dedup.deduplicate(iter(docs)))
        # Slightly different docs with threshold=1.0 should both be kept
        assert len(result) == 2

    def test_single_document(self):
        dedup = Deduplicator()
        docs = [_make_doc("Single document about legal matters.", doc_id="a")]
        result = list(dedup.deduplicate(iter(docs)))
        assert len(result) == 1
        assert dedup.stats.kept == 1
        assert dedup.stats.removed == 0

    def test_all_different_documents(self):
        dedup = Deduplicator(threshold=0.9)
        docs = [
            _make_doc(
                "The plaintiff filed a motion for summary judgment in the federal court.",
                doc_id="a",
            ),
            _make_doc(
                "Quantum entanglement enables instantaneous correlation between particles.",
                doc_id="b",
            ),
            _make_doc(
                "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
                doc_id="c",
            ),
        ]
        result = list(dedup.deduplicate(iter(docs)))
        assert len(result) == 3

    def test_near_duplicate_detection(self):
        """Two documents that are very similar but not identical."""
        dedup = Deduplicator(threshold=0.7, num_perm=128)
        base = "The court found the defendant liable for breach of contract under state law. " * 5
        near_dup = base + " Additional minor text here."
        docs = [
            _make_doc(base, doc_id="original"),
            _make_doc(near_dup, doc_id="near_dup"),
        ]
        result = list(dedup.deduplicate(iter(docs)))
        assert len(result) == 1
        assert result[0].doc_id == "original"

    def test_stats_summary_after_dedup(self):
        dedup = Deduplicator(threshold=0.8)
        text = "Repeated document about intellectual property law and patent infringement."
        docs = [_make_doc(text, doc_id=str(i)) for i in range(3)]
        list(dedup.deduplicate(iter(docs)))
        summary = dedup.stats.summary()
        assert summary["total_input"] == 3
        assert summary["kept"] == 1
        assert summary["removed"] == 2
        assert summary["dedup_rate"] == approx(2 / 3, abs=0.01)
