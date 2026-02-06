"""Abstract base loader for domain document ingestion.

All domain-specific loaders inherit from BaseLoader and implement the
load() method to yield Document objects from their respective sources.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """A single document from a domain corpus.

    Attributes:
        text: The raw document text.
        source: Where this document came from (e.g. "pile-of-law", "local").
        doc_id: Unique identifier for the document.
        metadata: Arbitrary key-value metadata (jurisdiction, date, category, etc.).
    """

    text: str
    source: str = ""
    doc_id: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def char_length(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class BaseLoader(ABC):
    """Abstract base class for all document loaders.

    Subclasses must implement `load()` which yields Document objects.
    """

    def __init__(self, source_name: str, **kwargs):
        self.source_name = source_name
        self.kwargs = kwargs

    @abstractmethod
    def load(self) -> Iterator[Document]:
        """Yield documents from the data source."""

    def load_all(self) -> list[Document]:
        """Load all documents into memory. Use with caution on large corpora."""
        return list(self.load())


class JSONLLoader(BaseLoader):
    """Load documents from a JSONL file.

    Each line must be a JSON object with at least a "text" field.
    Optional fields: "doc_id", "source", "metadata" (or any extra keys go into metadata).
    """

    def __init__(self, path: str | Path, source_name: str = "local", **kwargs):
        super().__init__(source_name=source_name, **kwargs)
        self.path = Path(path)

    def load(self) -> Iterator[Document]:
        with open(self.path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.pop("text", "")
                doc_id = obj.pop("doc_id", str(i))
                source = obj.pop("source", self.source_name)
                metadata = obj.pop("metadata", {})
                # Any remaining keys go into metadata
                metadata.update(obj)
                yield Document(
                    text=text,
                    source=source,
                    doc_id=doc_id,
                    metadata=metadata,
                )
