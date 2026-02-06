"""Domain-aware chat formatting for training data.

Converts raw documents into chat-template format suitable for
instruction tuning. Supports system prompts, multi-turn conversations,
and domain-specific formatting conventions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src.data.loaders.base_loader import Document

logger = logging.getLogger(__name__)

# Domain-specific system prompts
DOMAIN_SYSTEM_PROMPTS = {
    "legal": (
        "You are a legal assistant with expertise in contract analysis, case law, "
        "and legal reasoning. Provide accurate, well-cited legal analysis. "
        "Always note that your responses are for informational purposes only "
        "and do not constitute legal advice."
    ),
    "medical": (
        "You are a medical knowledge assistant with expertise in clinical medicine. "
        "Provide accurate, evidence-based medical information. "
        "Always recommend consulting a healthcare professional for medical decisions."
    ),
    "financial": (
        "You are a financial analysis assistant with expertise in financial markets, "
        "accounting, and investment analysis. Provide accurate financial information. "
        "Note that your responses do not constitute financial advice."
    ),
    "scientific": (
        "You are a scientific research assistant with expertise in academic literature. "
        "Provide accurate, well-cited scientific information."
    ),
}


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatExample:
    """A complete chat example for training."""

    messages: list[ChatMessage]
    doc_id: str = ""
    metadata: dict | None = None

    def to_dict(self) -> dict:
        result = {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
        }
        if self.doc_id:
            result["doc_id"] = self.doc_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class DomainChatFormatter:
    """Format raw documents into chat-template training examples.

    Supports two modes:
    - "completion": Document text becomes an assistant completion (continued pretraining style)
    - "qa": Document is framed as a user question + assistant answer pair

    Args:
        domain: Domain name for system prompt selection.
        mode: Formatting mode ("completion" or "qa").
        system_prompt: Custom system prompt (overrides domain default).
        max_length: Maximum character length for output text.
        chunk_overlap: Character overlap between chunks when splitting long documents.
    """

    def __init__(
        self,
        domain: str = "legal",
        mode: str = "completion",
        system_prompt: str | None = None,
        max_length: int = 8000,
        chunk_overlap: int = 200,
    ):
        self.domain = domain
        self.mode = mode
        self.system_prompt = system_prompt or DOMAIN_SYSTEM_PROMPTS.get(domain, "")
        self.max_length = max_length
        self.chunk_overlap = chunk_overlap

    def _chunk_text(self, text: str) -> list[str]:
        """Split long text into overlapping chunks."""
        if len(text) <= self.max_length:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.max_length
            # Try to break at a sentence boundary
            if end < len(text):
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                break_point = max(last_period, last_newline)
                if break_point > start + self.max_length // 2:
                    end = break_point + 1

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]

    def _format_completion(self, doc: Document) -> list[ChatExample]:
        """Format document as completion-style training data."""
        chunks = self._chunk_text(doc.text)
        examples = []

        for i, chunk in enumerate(chunks):
            messages = []
            if self.system_prompt:
                messages.append(ChatMessage(role="system", content=self.system_prompt))
            messages.append(ChatMessage(role="user", content="Continue the following text:"))
            messages.append(ChatMessage(role="assistant", content=chunk))

            doc_id = f"{doc.doc_id}:chunk_{i}" if len(chunks) > 1 else doc.doc_id
            examples.append(ChatExample(messages=messages, doc_id=doc_id, metadata=doc.metadata))

        return examples

    def _format_qa(self, doc: Document) -> list[ChatExample]:
        """Format document as Q&A-style training data.

        Uses the document title/metadata as a question prompt and the
        content as the answer.
        """
        title = doc.metadata.get("title", "")
        category = doc.metadata.get("category", doc.metadata.get("type", ""))

        if title:
            question = f"Explain the following {category} document: {title}"
        elif category:
            question = f"Provide analysis of this {category} document."
        else:
            question = f"Analyze the following {self.domain} document."

        chunks = self._chunk_text(doc.text)
        examples = []

        for i, chunk in enumerate(chunks):
            messages = []
            if self.system_prompt:
                messages.append(ChatMessage(role="system", content=self.system_prompt))

            q = question if i == 0 else f"{question} (continued, part {i + 1})"
            messages.append(ChatMessage(role="user", content=q))
            messages.append(ChatMessage(role="assistant", content=chunk))

            doc_id = f"{doc.doc_id}:chunk_{i}" if len(chunks) > 1 else doc.doc_id
            examples.append(ChatExample(messages=messages, doc_id=doc_id, metadata=doc.metadata))

        return examples

    def format(self, documents: Iterator[Document]) -> Iterator[ChatExample]:
        """Format a stream of documents into chat examples."""
        formatter = self._format_qa if self.mode == "qa" else self._format_completion

        for doc in documents:
            examples = formatter(doc)
            yield from examples

    def save_jsonl(self, examples: Iterator[ChatExample], output_path: str | Path) -> int:
        """Save chat examples to a JSONL file. Returns count of examples written."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")
                count += 1

        logger.info("Saved %d chat examples to %s", count, output_path)
        return count
