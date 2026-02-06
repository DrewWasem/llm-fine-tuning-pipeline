"""Near-duplicate removal using MinHash LSH.

Computes MinHash signatures for each document's n-gram shingles,
then uses Locality-Sensitive Hashing to efficiently find and remove
near-duplicate documents above a configurable similarity threshold.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from collections.abc import Iterator
from dataclasses import dataclass

from src.data.loaders.base_loader import Document

logger = logging.getLogger(__name__)

# Large prime for hash computation
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _ngrams(text: str, n: int = 5) -> set[str]:
    """Extract character n-grams (shingles) from text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _hash_shingle(shingle: str) -> int:
    """Hash a shingle to a 32-bit integer."""
    return struct.unpack("<I", hashlib.md5(shingle.encode("utf-8")).digest()[:4])[0]


class MinHashSignature:
    """Compute a MinHash signature for a document.

    Args:
        num_perm: Number of permutation hash functions (higher = more accurate, slower).
    """

    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        # Pre-generate random hash function parameters
        import random

        rng = random.Random(42)
        self._a = [rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
        self._b = [rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(num_perm)]

    def compute(self, text: str, ngram_size: int = 5) -> list[int]:
        """Compute MinHash signature for the given text."""
        shingles = _ngrams(text, ngram_size)
        if not shingles:
            return [_MAX_HASH] * self.num_perm

        hashed_shingles = [_hash_shingle(s) for s in shingles]

        signature = []
        for i in range(self.num_perm):
            min_hash = min(
                ((self._a[i] * h + self._b[i]) % _MERSENNE_PRIME) & _MAX_HASH
                for h in hashed_shingles
            )
            signature.append(min_hash)

        return signature

    @staticmethod
    def jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if len(sig_a) != len(sig_b):
            raise ValueError("Signatures must have the same length")
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)


class LSHIndex:
    """Locality-Sensitive Hashing index for fast approximate nearest-neighbor search.

    Divides MinHash signatures into bands. Documents that share a band hash
    are candidate duplicates.

    Args:
        num_bands: Number of bands to split the signature into.
        num_rows: Number of rows per band (num_bands * num_rows should equal num_perm).
    """

    def __init__(self, num_bands: int = 16, num_rows: int = 8):
        self.num_bands = num_bands
        self.num_rows = num_rows
        # Each band has a dict mapping band_hash -> set of doc_ids
        self._buckets: list[dict[int, set[str]]] = [
            {} for _ in range(num_bands)
        ]

    def insert(self, doc_id: str, signature: list[int]) -> set[str]:
        """Insert a signature and return the set of candidate duplicate doc_ids."""
        candidates: set[str] = set()

        for band_idx in range(self.num_bands):
            start = band_idx * self.num_rows
            end = start + self.num_rows
            band = tuple(signature[start:end])
            band_hash = hash(band)

            bucket = self._buckets[band_idx]
            if band_hash in bucket:
                candidates.update(bucket[band_hash])
                bucket[band_hash].add(doc_id)
            else:
                bucket[band_hash] = {doc_id}

        return candidates


@dataclass
class DedupStats:
    """Statistics from deduplication."""

    total_input: int = 0
    kept: int = 0
    removed: int = 0

    @property
    def dedup_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.removed / self.total_input

    def summary(self) -> dict:
        return {
            "total_input": self.total_input,
            "kept": self.kept,
            "removed": self.removed,
            "dedup_rate": round(self.dedup_rate, 4),
        }


class Deduplicator:
    """Remove near-duplicate documents from a stream using MinHash LSH.

    Args:
        threshold: Jaccard similarity threshold above which documents are
                   considered duplicates (0.0 to 1.0). Default 0.85.
        num_perm: Number of MinHash permutations. Higher = more accurate.
        num_bands: Number of LSH bands. Tune with num_rows for desired threshold.
        num_rows: Number of rows per LSH band.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        num_perm: int = 128,
        num_bands: int = 16,
        num_rows: int = 8,
    ):
        self.threshold = threshold
        self.minhash = MinHashSignature(num_perm=num_perm)
        self.lsh = LSHIndex(num_bands=num_bands, num_rows=num_rows)
        self.stats = DedupStats()
        # Store signatures for candidate verification
        self._signatures: dict[str, list[int]] = {}

    def _is_duplicate(self, doc_id: str, signature: list[int]) -> bool:
        """Check if a document is a near-duplicate of any previously seen document."""
        candidates = self.lsh.insert(doc_id, signature)

        for candidate_id in candidates:
            if candidate_id == doc_id:
                continue
            candidate_sig = self._signatures.get(candidate_id)
            if candidate_sig is None:
                continue
            similarity = MinHashSignature.jaccard_estimate(signature, candidate_sig)
            if similarity >= self.threshold:
                return True

        return False

    def deduplicate(self, documents: Iterator[Document]) -> Iterator[Document]:
        """Remove near-duplicates from a document stream."""
        for doc in documents:
            self.stats.total_input += 1
            doc_id = doc.doc_id or str(self.stats.total_input)

            signature = self.minhash.compute(doc.text)

            if self._is_duplicate(doc_id, signature):
                self.stats.removed += 1
                continue

            self._signatures[doc_id] = signature
            self.stats.kept += 1
            yield doc

        logger.info("Dedup stats: %s", self.stats.summary())
