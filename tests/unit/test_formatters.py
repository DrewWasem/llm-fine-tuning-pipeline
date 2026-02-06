"""Tests for data formatting (domain chat formatter)."""

from src.data.formatters.domain_chat import (
    DOMAIN_SYSTEM_PROMPTS,
    ChatExample,
    ChatMessage,
    DomainChatFormatter,
)
from src.data.loaders.base_loader import Document


def _make_doc(text: str = "", doc_id: str = "test_0", **kwargs) -> Document:
    default_text = "The court held that the defendant breached the contract. " * 10
    return Document(text=text or default_text, source="test", doc_id=doc_id, **kwargs)


# ── ChatExample tests ──────────────────────────────────────────────────────────


class TestChatExample:
    def test_to_dict(self):
        example = ChatExample(
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!"),
            ],
            doc_id="test_0",
        )
        d = example.to_dict()
        assert len(d["messages"]) == 3
        assert d["messages"][0]["role"] == "system"
        assert d["doc_id"] == "test_0"


# ── DomainChatFormatter tests ─────────────────────────────────────────────────


class TestDomainChatFormatter:
    def test_completion_mode(self):
        formatter = DomainChatFormatter(domain="legal", mode="completion")
        docs = [_make_doc()]
        examples = list(formatter.format(iter(docs)))
        assert len(examples) >= 1
        # Should have system + user + assistant messages
        msgs = examples[0].messages
        roles = [m.role for m in msgs]
        assert "system" in roles
        assert "assistant" in roles

    def test_qa_mode(self):
        formatter = DomainChatFormatter(domain="legal", mode="qa")
        docs = [_make_doc(metadata={"title": "Smith v Jones", "category": "opinion"})]
        examples = list(formatter.format(iter(docs)))
        assert len(examples) >= 1
        user_msg = next(m for m in examples[0].messages if m.role == "user")
        assert "opinion" in user_msg.content.lower() or "Smith v Jones" in user_msg.content

    def test_system_prompt_from_domain(self):
        formatter = DomainChatFormatter(domain="legal", mode="completion")
        docs = [_make_doc()]
        examples = list(formatter.format(iter(docs)))
        system_msg = next(m for m in examples[0].messages if m.role == "system")
        assert "legal" in system_msg.content.lower()

    def test_custom_system_prompt(self):
        formatter = DomainChatFormatter(domain="legal", system_prompt="Custom prompt")
        docs = [_make_doc()]
        examples = list(formatter.format(iter(docs)))
        system_msg = next(m for m in examples[0].messages if m.role == "system")
        assert system_msg.content == "Custom prompt"

    def test_chunking_long_document(self):
        long_text = "The court held that the defendant is liable. " * 1000
        formatter = DomainChatFormatter(domain="legal", max_length=500)
        docs = [_make_doc(text=long_text)]
        examples = list(formatter.format(iter(docs)))
        assert len(examples) > 1
        # Each chunk's doc_id should be distinct
        doc_ids = [e.doc_id for e in examples]
        assert len(set(doc_ids)) == len(doc_ids)

    def test_short_document_no_chunking(self):
        formatter = DomainChatFormatter(domain="legal", max_length=10000)
        docs = [_make_doc()]
        examples = list(formatter.format(iter(docs)))
        assert len(examples) == 1

    def test_save_jsonl(self, tmp_path):
        formatter = DomainChatFormatter(domain="legal", mode="completion")
        docs = [_make_doc(), _make_doc(doc_id="test_1")]
        examples = formatter.format(iter(docs))

        path = tmp_path / "output.jsonl"
        count = formatter.save_jsonl(examples, path)
        assert count >= 2
        assert path.exists()

        import json

        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == count
        assert "messages" in lines[0]

    def test_all_domains_have_system_prompts(self):
        for domain in ["legal", "medical", "financial", "scientific"]:
            assert domain in DOMAIN_SYSTEM_PROMPTS

    def test_empty_stream(self):
        formatter = DomainChatFormatter(domain="legal")
        examples = list(formatter.format(iter([])))
        assert len(examples) == 0
