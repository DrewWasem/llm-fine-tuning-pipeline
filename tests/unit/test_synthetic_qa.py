"""Tests for synthetic Q&A generation."""

import json
from pathlib import Path

from src.corpus.synthetic.qa_generator import (
    DOMAIN_TEMPLATES,
    QAGenerator,
    QAPair,
    _extract_passages,
    _has_enough_content,
    load_qa_pairs,
    save_qa_pairs,
)
from src.data.loaders.base_loader import Document


def _make_doc(text: str = "", doc_id: str = "test_0", **kwargs) -> Document:
    return Document(
        text=text or ("This is a legal case regarding contract law. " * 20),
        source="test",
        doc_id=doc_id,
        metadata=kwargs.get("metadata", {}),
    )


# ── Passage extraction tests ────────────────────────────────────────────────


class TestExtractPassages:
    def test_short_text_single_passage(self):
        text = "Short legal text."
        passages = _extract_passages(text, passage_length=1000)
        assert len(passages) == 1
        assert passages[0] == "Short legal text."

    def test_long_text_multiple_passages(self):
        text = "Legal principle. " * 200  # ~3400 chars
        passages = _extract_passages(text, passage_length=500)
        assert len(passages) >= 2

    def test_empty_text(self):
        assert _extract_passages("") == []
        assert _extract_passages("   ") == []

    def test_passage_overlap(self):
        text = "Word " * 1000  # 5000 chars
        passages = _extract_passages(text, passage_length=1000, overlap=200)
        assert len(passages) >= 2
        # With overlap, later passages should start before previous ones end
        # We just check we get reasonable chunking
        for p in passages:
            assert len(p) > 0


class TestHasEnoughContent:
    def test_enough_words(self):
        assert _has_enough_content("This is a sentence with enough words to pass the test easily.", min_words=5)

    def test_too_few_words(self):
        assert not _has_enough_content("Too short.", min_words=30)

    def test_mostly_numbers(self):
        assert not _has_enough_content("123 456 789 " * 10, min_words=5)


# ── QAPair tests ─────────────────────────────────────────────────────────────


class TestQAPair:
    def test_to_dict(self):
        pair = QAPair(
            question="What is consideration?",
            answer="Something of value exchanged.",
            source_doc_id="doc_1",
            category="qa",
        )
        d = pair.to_dict()
        assert d["question"] == "What is consideration?"
        assert d["answer"] == "Something of value exchanged."
        assert d["source_doc_id"] == "doc_1"
        assert d["category"] == "qa"

    def test_to_chat_messages_without_system(self):
        pair = QAPair(question="Q?", answer="A.")
        messages = pair.to_chat_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_to_chat_messages_with_system(self):
        pair = QAPair(question="Q?", answer="A.")
        messages = pair.to_chat_messages(system_prompt="You are a legal assistant.")
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"


# ── QAGenerator tests ────────────────────────────────────────────────────────


class TestQAGenerator:
    def test_generate_from_document(self):
        gen = QAGenerator(domain="legal", max_pairs_per_doc=3, seed=42)
        doc = _make_doc("This is a substantial legal document about contract law. " * 30)
        pairs = gen.generate_from_document(doc)
        assert len(pairs) > 0
        assert len(pairs) <= 3
        for pair in pairs:
            assert pair.question
            assert pair.answer
            assert pair.source_doc_id == "test_0"
            assert pair.category == "qa"

    def test_skip_short_documents(self):
        gen = QAGenerator(domain="legal", min_doc_length=500)
        doc = _make_doc("Short text.")
        pairs = gen.generate_from_document(doc)
        assert pairs == []

    def test_max_pairs_per_doc(self):
        gen = QAGenerator(domain="legal", max_pairs_per_doc=1, seed=42)
        doc = _make_doc("This is a long legal document. " * 100)
        pairs = gen.generate_from_document(doc)
        assert len(pairs) <= 1

    def test_generate_from_documents(self):
        gen = QAGenerator(domain="legal", max_pairs_per_doc=2, seed=42)
        docs = [
            _make_doc("Legal document about torts and liability. " * 30, doc_id=f"doc_{i}")
            for i in range(5)
        ]
        pairs = list(gen.generate_from_documents(iter(docs), max_pairs=6))
        assert 0 < len(pairs) <= 6

    def test_max_pairs_limit(self):
        gen = QAGenerator(domain="legal", max_pairs_per_doc=5, seed=42)
        docs = [
            _make_doc("Legal text about contracts and obligations. " * 50, doc_id=f"doc_{i}")
            for i in range(10)
        ]
        pairs = list(gen.generate_from_documents(iter(docs), max_pairs=3))
        assert len(pairs) <= 3

    def test_domain_templates(self):
        assert "legal" in DOMAIN_TEMPLATES
        assert "medical" in DOMAIN_TEMPLATES
        assert len(DOMAIN_TEMPLATES["legal"]) > 5

    def test_custom_templates(self):
        templates = ["What does this mean?\n\n{passage}"]
        gen = QAGenerator(templates=templates, seed=42)
        doc = _make_doc("Legal text about court proceedings. " * 30)
        pairs = gen.generate_from_document(doc)
        for pair in pairs:
            assert "What does this mean?" in pair.question

    def test_reproducibility(self):
        doc = _make_doc("Legal text about jurisdiction and venue. " * 30)
        gen1 = QAGenerator(seed=42)
        gen2 = QAGenerator(seed=42)
        pairs1 = gen1.generate_from_document(doc)
        pairs2 = gen2.generate_from_document(doc)
        assert len(pairs1) == len(pairs2)
        for p1, p2 in zip(pairs1, pairs2):
            assert p1.question == p2.question


# ── Save/Load tests ──────────────────────────────────────────────────────────


class TestSaveLoadQA:
    def test_save_chat_format(self, tmp_path: Path):
        pairs = [
            QAPair(question="Q1?", answer="A1.", source_doc_id="d1"),
            QAPair(question="Q2?", answer="A2.", source_doc_id="d2"),
        ]
        path = tmp_path / "qa.jsonl"
        count = save_qa_pairs(iter(pairs), path, system_prompt="System.", fmt="chat")
        assert count == 2

        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert "messages" in lines[0]
        assert lines[0]["messages"][0]["role"] == "system"

    def test_save_raw_format(self, tmp_path: Path):
        pairs = [QAPair(question="Q?", answer="A.", source_doc_id="d1")]
        path = tmp_path / "qa.jsonl"
        count = save_qa_pairs(iter(pairs), path, fmt="raw")
        assert count == 1

        with open(path) as f:
            data = json.loads(f.readline())
        assert data["question"] == "Q?"
        assert data["answer"] == "A."

    def test_load_qa_pairs(self, tmp_path: Path):
        path = tmp_path / "qa.jsonl"
        records = [
            {"question": "Q1?", "answer": "A1.", "source_doc_id": "d1", "category": "qa"},
            {"question": "Q2?", "answer": "A2."},
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        pairs = load_qa_pairs(path)
        assert len(pairs) == 2
        assert pairs[0].question == "Q1?"
        assert pairs[0].source_doc_id == "d1"
        assert pairs[1].source_doc_id == ""
