"""Legal document loader.

Loads legal documents from HuggingFace datasets (e.g. pile-of-law)
or from local files. Supports filtering by category and jurisdiction.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from src.data.loaders.base_loader import BaseLoader, Document

logger = logging.getLogger(__name__)


class HuggingFaceLoader(BaseLoader):
    """Load documents from a HuggingFace dataset.

    Uses the `datasets` library to stream documents, avoiding
    loading the entire dataset into memory.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. "pile-of-law/pile-of-law").
        subset: Dataset subset/config name (e.g. "default").
        split: Dataset split (e.g. "train", "validation").
        text_field: Name of the field containing document text.
        max_documents: Maximum number of documents to load (None for all).
        streaming: Whether to use streaming mode (recommended for large datasets).
    """

    def __init__(
        self,
        dataset_name: str,
        subset: str | None = None,
        split: str = "train",
        text_field: str = "text",
        max_documents: int | None = None,
        streaming: bool = True,
        source_name: str = "",
        **kwargs,
    ):
        source = source_name or dataset_name
        super().__init__(source_name=source, **kwargs)
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.text_field = text_field
        self.max_documents = max_documents
        self.streaming = streaming

    def load(self) -> Iterator[Document]:
        from datasets import load_dataset

        logger.info(
            "Loading dataset %s (subset=%s, split=%s, streaming=%s)",
            self.dataset_name,
            self.subset,
            self.split,
            self.streaming,
        )

        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=self.streaming,
        )

        for i, row in enumerate(ds):
            if self.max_documents and i >= self.max_documents:
                break

            text = row.get(self.text_field, "")
            if not text:
                continue

            # Build metadata from all non-text fields
            metadata = {k: v for k, v in row.items() if k != self.text_field}

            yield Document(
                text=text,
                source=self.source_name,
                doc_id=f"{self.source_name}:{i}",
                metadata=metadata,
            )

        logger.info("Loaded %d documents from %s", i + 1 if 'i' in dir() else 0, self.dataset_name)


class LegalLoader(BaseLoader):
    """Load legal documents from configured sources.

    Supports multiple source types:
    - "huggingface": Load from HuggingFace datasets (e.g. pile-of-law)
    - "local": Load from local JSONL files

    Args:
        sources: List of source configurations, each with 'name', 'type',
                 and source-specific parameters.
        max_documents: Global cap on total documents across all sources.
        categories: Filter to only these document categories (if supported by source).
    """

    def __init__(
        self,
        sources: list[dict] | None = None,
        max_documents: int | None = None,
        categories: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(source_name="legal", **kwargs)
        self.sources = sources or []
        self.max_documents = max_documents
        self.categories = categories
        self._total_loaded = 0

    def _should_include(self, doc: Document) -> bool:
        """Check if a document matches the category filter."""
        if not self.categories:
            return True
        doc_category = doc.metadata.get("category", "")
        doc_type = doc.metadata.get("type", "")
        return doc_category in self.categories or doc_type in self.categories

    def _load_huggingface_source(self, source: dict) -> Iterator[Document]:
        """Load from a HuggingFace dataset source."""
        dataset_name = source.get("dataset", source.get("name", ""))
        loader = HuggingFaceLoader(
            dataset_name=dataset_name,
            subset=source.get("subset"),
            split=source.get("split", "train"),
            text_field=source.get("text_field", "text"),
            max_documents=source.get("max_documents"),
            streaming=source.get("streaming", True),
        )
        yield from loader.load()

    def _load_local_source(self, source: dict) -> Iterator[Document]:
        """Load from local JSONL files."""
        from src.data.loaders.base_loader import JSONLLoader

        path = source.get("path", "")
        if not path:
            logger.warning("Local source missing 'path', skipping")
            return

        loader = JSONLLoader(path=path, source_name=source.get("name", "local"))
        yield from loader.load()

    def load(self) -> Iterator[Document]:
        for source in self.sources:
            source_type = source.get("type", "huggingface")

            if source_type in ("huggingface", "hf"):
                doc_iter = self._load_huggingface_source(source)
            elif source_type in ("local", "jsonl"):
                doc_iter = self._load_local_source(source)
            else:
                logger.warning("Unknown source type: %s, skipping", source_type)
                continue

            for doc in doc_iter:
                if not self._should_include(doc):
                    continue

                if self.max_documents and self._total_loaded >= self.max_documents:
                    return

                self._total_loaded += 1
                yield doc

        logger.info("LegalLoader: loaded %d documents total", self._total_loaded)
