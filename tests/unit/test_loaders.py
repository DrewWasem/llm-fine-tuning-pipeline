"""Tests for data loaders."""

import json
from pathlib import Path

import pytest

from src.data.loaders.base_loader import Document, JSONLLoader
from src.data.loaders.legal_loader import LegalLoader

# ── Document tests ─────────────────────────────────────────────────────────────


class TestDocument:
    def test_char_length(self):
        doc = Document(text="hello world")
        assert doc.char_length == 11

    def test_word_count(self):
        doc = Document(text="hello world foo bar")
        assert doc.word_count == 4

    def test_empty_document(self):
        doc = Document(text="")
        assert doc.char_length == 0
        assert doc.word_count == 0

    def test_metadata(self):
        doc = Document(text="test", metadata={"category": "contract"})
        assert doc.metadata["category"] == "contract"


# ── JSONLLoader tests ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    """Create a sample JSONL file for testing."""
    path = tmp_path / "docs.jsonl"
    records = [
        {"text": "First document text", "doc_id": "doc_0", "category": "contract"},
        {"text": "Second document text", "doc_id": "doc_1", "category": "opinion"},
        {"text": "", "doc_id": "doc_2"},  # empty text
        {"text": "Third document text"},  # no doc_id
    ]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


class TestJSONLLoader:
    def test_load_basic(self, sample_jsonl: Path):
        loader = JSONLLoader(path=sample_jsonl)
        docs = loader.load_all()
        assert len(docs) == 4

    def test_load_text_content(self, sample_jsonl: Path):
        loader = JSONLLoader(path=sample_jsonl)
        docs = loader.load_all()
        assert docs[0].text == "First document text"
        assert docs[1].text == "Second document text"

    def test_load_doc_id(self, sample_jsonl: Path):
        loader = JSONLLoader(path=sample_jsonl)
        docs = loader.load_all()
        assert docs[0].doc_id == "doc_0"
        assert docs[3].doc_id == "3"  # auto-generated

    def test_load_extra_fields_go_to_metadata(self, sample_jsonl: Path):
        loader = JSONLLoader(path=sample_jsonl)
        docs = loader.load_all()
        assert docs[0].metadata["category"] == "contract"

    def test_load_source_name(self, sample_jsonl: Path):
        loader = JSONLLoader(path=sample_jsonl, source_name="test-source")
        docs = loader.load_all()
        assert docs[0].source == "test-source"

    def test_load_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        loader = JSONLLoader(path=path)
        docs = loader.load_all()
        assert len(docs) == 0


# ── LegalLoader tests ─────────────────────────────────────────────────────────


class TestLegalLoader:
    def test_load_from_local_source(self, sample_jsonl: Path):
        sources = [{"name": "test", "type": "local", "path": str(sample_jsonl)}]
        loader = LegalLoader(sources=sources)
        docs = loader.load_all()
        assert len(docs) == 4

    def test_max_documents_cap(self, sample_jsonl: Path):
        sources = [{"name": "test", "type": "local", "path": str(sample_jsonl)}]
        loader = LegalLoader(sources=sources, max_documents=2)
        docs = loader.load_all()
        assert len(docs) == 2

    def test_category_filter(self, sample_jsonl: Path):
        sources = [{"name": "test", "type": "local", "path": str(sample_jsonl)}]
        loader = LegalLoader(sources=sources, categories=["contract"])
        docs = loader.load_all()
        assert len(docs) == 1
        assert docs[0].metadata["category"] == "contract"

    def test_unknown_source_type_skipped(self):
        sources = [{"name": "bad", "type": "ftp"}]
        loader = LegalLoader(sources=sources)
        docs = loader.load_all()
        assert len(docs) == 0
