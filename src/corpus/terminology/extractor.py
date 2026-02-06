"""Domain terminology extraction using TF-IDF + spaCy NER hybrid approach.

Extracts domain-specific terms from a corpus by combining:
1. TF-IDF scoring to find statistically important terms
2. spaCy NER to identify named entities and noun phrases
3. Frequency filtering to keep only terms above a minimum threshold
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from src.config.settings import TerminologyConfig
from src.data.loaders.base_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Term:
    """A single domain term with its statistics."""

    text: str
    frequency: int = 0
    document_frequency: int = 0
    tfidf_score: float = 0.0
    pos_tags: list[str] = field(default_factory=list)
    is_entity: bool = False
    entity_label: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "frequency": self.frequency,
            "document_frequency": self.document_frequency,
            "tfidf_score": round(self.tfidf_score, 6),
            "pos_tags": self.pos_tags,
            "is_entity": self.is_entity,
            "entity_label": self.entity_label,
        }


def _tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer as fallback when spaCy is unavailable."""
    text = text.lower()
    # Keep hyphenated words, split on other non-alphanumeric
    tokens = re.findall(r"\b[a-z][a-z\-']*[a-z]\b|\b[a-z]\b", text)
    return tokens


def _extract_ngrams(tokens: list[str], min_n: int = 1, max_n: int = 3) -> list[str]:
    """Extract n-grams from a list of tokens."""
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams.append(ngram)
    return ngrams


class TerminologyExtractor:
    """Extract domain-specific terminology from a document corpus.

    Uses a hybrid approach combining TF-IDF statistics with optional
    spaCy NER for entity recognition.

    Args:
        config: Terminology configuration (method, min_frequency, pos_tags).
        use_spacy: Whether to use spaCy for NER. Falls back to TF-IDF only if False.
        max_ngram: Maximum n-gram size for term candidates.
        top_k: Maximum number of terms to return (by TF-IDF score).
    """

    def __init__(
        self,
        config: TerminologyConfig | None = None,
        use_spacy: bool = True,
        max_ngram: int = 3,
        top_k: int = 5000,
    ):
        self.config = config or TerminologyConfig()
        self.use_spacy = use_spacy
        self.max_ngram = max_ngram
        self.top_k = top_k

        self._nlp = None
        if use_spacy:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_lg")
                logger.info("Loaded spaCy model for NER")
            except (ImportError, OSError):
                logger.warning("spaCy model not available, falling back to TF-IDF only")
                self.use_spacy = False

    def _extract_terms_tfidf(self, documents: list[Document]) -> dict[str, Term]:
        """Extract terms using TF-IDF scoring."""
        # Count term frequencies and document frequencies
        term_freq: Counter[str] = Counter()
        doc_freq: Counter[str] = Counter()
        num_docs = len(documents)

        for doc in documents:
            tokens = _tokenize_simple(doc.text)
            ngrams = _extract_ngrams(tokens, min_n=1, max_n=self.max_ngram)

            # Term frequency
            doc_term_counts = Counter(ngrams)
            term_freq.update(doc_term_counts)

            # Document frequency (count each term once per document)
            doc_freq.update(doc_term_counts.keys())

        # Compute TF-IDF
        terms: dict[str, Term] = {}
        for term_text, freq in term_freq.items():
            if freq < self.config.min_frequency:
                continue
            # Skip single characters and very short terms
            if len(term_text) < 3:
                continue

            df = doc_freq[term_text]
            idf = math.log((num_docs + 1) / (df + 1)) + 1
            tfidf = freq * idf

            terms[term_text] = Term(
                text=term_text,
                frequency=freq,
                document_frequency=df,
                tfidf_score=tfidf,
            )

        return terms

    def _enrich_with_spacy(
        self, terms: dict[str, Term], documents: list[Document]
    ) -> dict[str, Term]:
        """Enrich terms with spaCy NER and POS tag information."""
        if not self._nlp:
            return terms

        allowed_pos = set(self.config.pos_tags) if self.config.pos_tags else None
        entity_terms: dict[str, Term] = {}

        for doc in documents:
            # Process a sample of each document to limit computation
            text_sample = doc.text[:5000]
            spacy_doc = self._nlp(text_sample)

            # Extract named entities
            for ent in spacy_doc.ents:
                ent_text = ent.text.lower().strip()
                if len(ent_text) < 3:
                    continue
                if ent_text in terms:
                    terms[ent_text].is_entity = True
                    terms[ent_text].entity_label = ent.label_
                elif ent_text not in entity_terms:
                    entity_terms[ent_text] = Term(
                        text=ent_text,
                        frequency=1,
                        is_entity=True,
                        entity_label=ent.label_,
                    )
                else:
                    entity_terms[ent_text].frequency += 1

            # Enrich with POS tags for noun phrases
            for chunk in spacy_doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                if chunk_text in terms:
                    pos = [token.pos_ for token in chunk]
                    terms[chunk_text].pos_tags = list(set(pos))

        # Merge entity terms that meet frequency threshold
        for text, term in entity_terms.items():
            if term.frequency >= self.config.min_frequency:
                terms[text] = term

        # Filter by POS tags if specified
        if allowed_pos:
            terms = {
                k: v for k, v in terms.items()
                if v.is_entity or not v.pos_tags or any(p in allowed_pos for p in v.pos_tags)
            }

        return terms

    def extract(self, documents: list[Document] | Iterator[Document]) -> list[Term]:
        """Extract domain terminology from a collection of documents.

        Args:
            documents: Documents to extract terminology from.

        Returns:
            List of Term objects sorted by TF-IDF score (descending).
        """
        if not isinstance(documents, list):
            documents = list(documents)

        if not documents:
            return []

        logger.info("Extracting terminology from %d documents", len(documents))

        # Step 1: TF-IDF extraction
        terms = self._extract_terms_tfidf(documents)
        logger.info("TF-IDF extracted %d candidate terms", len(terms))

        # Step 2: Enrich with spaCy NER (if available)
        if self.use_spacy:
            terms = self._enrich_with_spacy(terms, documents)
            logger.info("After spaCy enrichment: %d terms", len(terms))

        # Step 3: Sort by TF-IDF score and take top-k
        sorted_terms = sorted(terms.values(), key=lambda t: t.tfidf_score, reverse=True)
        return sorted_terms[: self.top_k]

    def save(self, terms: list[Term], output_path: str | Path) -> None:
        """Save extracted terms to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "num_terms": len(terms),
            "config": {
                "method": self.config.method,
                "min_frequency": self.config.min_frequency,
                "use_spacy": self.use_spacy,
                "max_ngram": self.max_ngram,
            },
            "terms": [t.to_dict() for t in terms],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved %d terms to %s", len(terms), output_path)
