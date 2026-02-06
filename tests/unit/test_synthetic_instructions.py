"""Tests for synthetic instruction generation."""

import json
from pathlib import Path

from src.corpus.synthetic.instruction_generator import (
    DOMAIN_INSTRUCTION_TEMPLATES,
    InstructionExample,
    InstructionGenerator,
    _split_into_passages,
    save_instructions,
)
from src.data.loaders.base_loader import Document


def _make_doc(text: str = "", doc_id: str = "test_0", **kwargs) -> Document:
    return Document(
        text=text or ("This is a legal document about contract law and obligations. " * 20),
        source="test",
        doc_id=doc_id,
        metadata=kwargs.get("metadata", {}),
    )


# ── Passage splitting tests ─────────────────────────────────────────────────


class TestSplitIntoPassages:
    def test_short_text(self):
        passages = _split_into_passages("Short text.", max_length=1000)
        assert len(passages) == 1

    def test_long_text(self):
        text = "Legal text about contracts. " * 200
        passages = _split_into_passages(text, max_length=500)
        assert len(passages) >= 2

    def test_empty_text(self):
        assert _split_into_passages("") == []
        assert _split_into_passages("   ") == []

    def test_filters_tiny_chunks(self):
        # Passages with fewer than 20 words should be filtered
        text = "A short bit. " + ("A much longer passage about legal matters. " * 50)
        passages = _split_into_passages(text, max_length=500)
        for p in passages:
            assert len(p.split()) >= 20


# ── InstructionExample tests ─────────────────────────────────────────────────


class TestInstructionExample:
    def test_to_dict(self):
        ex = InstructionExample(
            instruction="Summarize this text.",
            response="This is a summary.",
            task_type="summarize",
            source_doc_id="doc_1",
        )
        d = ex.to_dict()
        assert d["instruction"] == "Summarize this text."
        assert d["response"] == "This is a summary."
        assert d["task_type"] == "summarize"

    def test_to_chat_messages_without_system(self):
        ex = InstructionExample(instruction="Do this.", response="Done.")
        messages = ex.to_chat_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_to_chat_messages_with_system(self):
        ex = InstructionExample(instruction="Do this.", response="Done.")
        messages = ex.to_chat_messages(system_prompt="Be helpful.")
        assert len(messages) == 3
        assert messages[0]["role"] == "system"


# ── InstructionGenerator tests ───────────────────────────────────────────────


class TestInstructionGenerator:
    def test_generate_from_document(self):
        gen = InstructionGenerator(domain="legal", max_examples_per_doc=3, seed=42)
        doc = _make_doc("This is a substantial legal document about contract law. " * 30)
        examples = gen.generate_from_document(doc)
        assert len(examples) > 0
        assert len(examples) <= 3
        for ex in examples:
            assert ex.instruction
            assert ex.response
            assert ex.source_doc_id == "test_0"
            assert ex.task_type in ("summarize", "analyze", "explain", "classify", "extract", "compare", "draft")

    def test_skip_short_documents(self):
        gen = InstructionGenerator(min_doc_length=500)
        doc = _make_doc("Short.")
        assert gen.generate_from_document(doc) == []

    def test_max_examples_per_doc(self):
        gen = InstructionGenerator(max_examples_per_doc=1, seed=42)
        doc = _make_doc("Legal text about torts. " * 100)
        examples = gen.generate_from_document(doc)
        assert len(examples) <= 1

    def test_generate_from_documents(self):
        gen = InstructionGenerator(max_examples_per_doc=2, seed=42)
        docs = [
            _make_doc("Legal document about liability. " * 30, doc_id=f"doc_{i}")
            for i in range(5)
        ]
        examples = list(gen.generate_from_documents(iter(docs), max_examples=4))
        assert 0 < len(examples) <= 4

    def test_generate_term_definitions(self):
        terms = ["fiduciary duty", "breach of contract", "estoppel"]
        gen = InstructionGenerator(terminology=terms, seed=42)
        definitions = gen.generate_term_definitions()
        assert len(definitions) == 3
        for d in definitions:
            assert d.task_type == "definition"
            assert any(t in d.instruction for t in terms)

    def test_generate_term_definitions_max_terms(self):
        terms = ["term1", "term2", "term3", "term4", "term5"]
        gen = InstructionGenerator(terminology=terms, seed=42)
        definitions = gen.generate_term_definitions(max_terms=2)
        assert len(definitions) == 2

    def test_generate_term_definitions_no_terms(self):
        gen = InstructionGenerator(terminology=[])
        assert gen.generate_term_definitions() == []

    def test_domain_templates(self):
        assert "legal" in DOMAIN_INSTRUCTION_TEMPLATES
        assert "medical" in DOMAIN_INSTRUCTION_TEMPLATES
        assert len(DOMAIN_INSTRUCTION_TEMPLATES["legal"]) >= 10

    def test_custom_templates(self):
        templates = [("custom", "Custom instruction: {passage}")]
        gen = InstructionGenerator(templates=templates, seed=42)
        doc = _make_doc("Legal text about court proceedings. " * 30)
        examples = gen.generate_from_document(doc)
        for ex in examples:
            assert "Custom instruction:" in ex.instruction
            assert ex.task_type == "custom"

    def test_reproducibility(self):
        doc = _make_doc("Legal text about jurisdiction. " * 30)
        gen1 = InstructionGenerator(seed=42)
        gen2 = InstructionGenerator(seed=42)
        ex1 = gen1.generate_from_document(doc)
        ex2 = gen2.generate_from_document(doc)
        assert len(ex1) == len(ex2)
        for a, b in zip(ex1, ex2):
            assert a.instruction == b.instruction


# ── Save tests ───────────────────────────────────────────────────────────────


class TestSaveInstructions:
    def test_save_chat_format(self, tmp_path: Path):
        examples = [
            InstructionExample(instruction="Do X.", response="Done X."),
            InstructionExample(instruction="Do Y.", response="Done Y."),
        ]
        path = tmp_path / "inst.jsonl"
        count = save_instructions(iter(examples), path, system_prompt="System.", fmt="chat")
        assert count == 2

        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert "messages" in lines[0]
        assert lines[0]["messages"][0]["role"] == "system"

    def test_save_raw_format(self, tmp_path: Path):
        examples = [InstructionExample(instruction="Explain.", response="Here.")]
        path = tmp_path / "inst.jsonl"
        count = save_instructions(iter(examples), path, fmt="raw")
        assert count == 1

        with open(path) as f:
            data = json.loads(f.readline())
        assert data["instruction"] == "Explain."
        assert data["response"] == "Here."
