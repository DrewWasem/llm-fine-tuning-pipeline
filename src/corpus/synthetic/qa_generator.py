"""Generate Q&A pairs from domain documents.

Supports two modes:
- Template-based: Uses predefined question templates + document passages (no LLM needed)
- Model-based: Uses a loaded LLM to generate questions from document passages

Output is compatible with the DomainChatFormatter chat-message format.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from src.data.loaders.base_loader import Document

logger = logging.getLogger(__name__)

# Domain-specific question templates.  Each template expects the document
# passage to be provided as context; the answer is extracted from the passage.
LEGAL_QA_TEMPLATES = [
    "What is the main legal principle discussed in the following passage?",
    "Summarize the key legal arguments presented in this text.",
    "What are the legal implications described in this passage?",
    "Identify the relevant legal standards referenced in this text.",
    "What legal reasoning is applied in the following passage?",
    "What obligations or duties are established in this text?",
    "Explain the jurisdictional considerations mentioned in this passage.",
    "What remedies or outcomes are discussed in this text?",
    "Identify any statutory references or case citations in this passage.",
    "What are the key contractual terms discussed in this text?",
]

MEDICAL_QA_TEMPLATES = [
    "What medical condition or diagnosis is discussed in this passage?",
    "Summarize the treatment approach described in this text.",
    "What are the key clinical findings mentioned in this passage?",
    "Identify the relevant medical terminology in this text.",
    "What are the risk factors discussed in this passage?",
]

FINANCIAL_QA_TEMPLATES = [
    "What financial metrics are discussed in this passage?",
    "Summarize the investment thesis presented in this text.",
    "What market trends are described in this passage?",
    "Identify the key financial risks mentioned in this text.",
    "What regulatory considerations are discussed in this passage?",
]

DOMAIN_TEMPLATES: dict[str, list[str]] = {
    "legal": LEGAL_QA_TEMPLATES,
    "medical": MEDICAL_QA_TEMPLATES,
    "financial": FINANCIAL_QA_TEMPLATES,
}


@dataclass
class QAPair:
    """A single question-answer pair generated from a source document."""

    question: str
    answer: str
    source_doc_id: str = ""
    category: str = "qa"
    quality_score: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_chat_messages(self, system_prompt: str = "") -> list[dict]:
        """Convert to chat-message format for training."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.question})
        messages.append({"role": "assistant", "content": self.answer})
        return messages

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "source_doc_id": self.source_doc_id,
            "category": self.category,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }


def _extract_passages(text: str, passage_length: int = 1500, overlap: int = 200) -> list[str]:
    """Split document text into passages suitable for Q&A generation."""
    if len(text) <= passage_length:
        return [text.strip()] if text.strip() else []

    passages = []
    start = 0
    while start < len(text):
        end = start + passage_length
        if end < len(text):
            # Break at sentence boundary
            for delim in (".\n", ". ", "\n\n", "\n"):
                pos = text.rfind(delim, start + passage_length // 2, end)
                if pos > start:
                    end = pos + len(delim)
                    break
        chunk = text[start:end].strip()
        if chunk:
            passages.append(chunk)
        start = end - overlap

    return passages


def _has_enough_content(passage: str, min_words: int = 30) -> bool:
    """Check if a passage has enough content for meaningful Q&A."""
    words = passage.split()
    if len(words) < min_words:
        return False
    # Reject passages that are mostly numbers/punctuation
    alpha_ratio = sum(1 for c in passage if c.isalpha()) / max(len(passage), 1)
    return alpha_ratio > 0.5


class QAGenerator:
    """Generate Q&A pairs from domain documents.

    Args:
        domain: Domain name for template selection.
        templates: Custom question templates (overrides domain defaults).
        passage_length: Target character length for document passages.
        min_doc_length: Minimum document length to generate Q&A from.
        max_pairs_per_doc: Maximum Q&A pairs to generate per document.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        domain: str = "legal",
        templates: list[str] | None = None,
        passage_length: int = 1500,
        min_doc_length: int = 200,
        max_pairs_per_doc: int = 5,
        seed: int = 42,
    ):
        self.domain = domain
        self.templates = templates or DOMAIN_TEMPLATES.get(domain, LEGAL_QA_TEMPLATES)
        self.passage_length = passage_length
        self.min_doc_length = min_doc_length
        self.max_pairs_per_doc = max_pairs_per_doc
        self.rng = random.Random(seed)

    def generate_from_document(self, doc: Document) -> list[QAPair]:
        """Generate Q&A pairs from a single document using templates.

        Each pair uses a passage from the document as the answer and a
        randomly selected template as the question.
        """
        if len(doc.text) < self.min_doc_length:
            return []

        passages = _extract_passages(doc.text, passage_length=self.passage_length)
        passages = [p for p in passages if _has_enough_content(p)]

        if not passages:
            return []

        pairs = []
        selected = self.rng.sample(passages, min(len(passages), self.max_pairs_per_doc))

        for passage in selected:
            template = self.rng.choice(self.templates)
            question = f"{template}\n\nContext:\n{passage}"
            pair = QAPair(
                question=question,
                answer=passage,
                source_doc_id=doc.doc_id,
                category="qa",
                metadata={"template": template, "domain": self.domain},
            )
            pairs.append(pair)

        return pairs

    def generate_from_documents(
        self, documents: Iterator[Document], max_pairs: int | None = None
    ) -> Iterator[QAPair]:
        """Generate Q&A pairs from a stream of documents.

        Args:
            documents: Iterator of Document objects.
            max_pairs: Maximum total pairs to generate (None = unlimited).
        """
        count = 0
        for doc in documents:
            if max_pairs and count >= max_pairs:
                break
            pairs = self.generate_from_document(doc)
            for pair in pairs:
                if max_pairs and count >= max_pairs:
                    break
                yield pair
                count += 1

        logger.info("Generated %d Q&A pairs", count)

    def generate_with_model(
        self, doc: Document, model, tokenizer, max_new_tokens: int = 256
    ) -> list[QAPair]:
        """Use an LLM to generate questions from document passages.

        The model generates questions about the passage, and the passage
        serves as the reference answer.
        """
        from src.evaluation.base import generate_text

        passages = _extract_passages(doc.text, passage_length=self.passage_length)
        passages = [p for p in passages if _has_enough_content(p)]

        if not passages:
            return []

        pairs = []
        selected = self.rng.sample(passages, min(len(passages), self.max_pairs_per_doc))

        for passage in selected:
            prompt = (
                f"You are an expert {self.domain} instructor. "
                f"Read the following passage and generate a clear, specific question "
                f"that can be answered using the information in the passage.\n\n"
                f"Passage:\n{passage[:800]}\n\n"
                f"Question:"
            )
            generated_question = generate_text(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens
            )

            if generated_question and len(generated_question) > 10:
                # Clean up the generated question
                generated_question = generated_question.strip()
                if not generated_question.endswith("?"):
                    generated_question += "?"

                pair = QAPair(
                    question=generated_question,
                    answer=passage,
                    source_doc_id=doc.doc_id,
                    category="qa_model_generated",
                    metadata={"domain": self.domain, "generation_method": "llm"},
                )
                pairs.append(pair)

        return pairs


def save_qa_pairs(
    pairs: Iterator[QAPair],
    output_path: str | Path,
    system_prompt: str = "",
    fmt: str = "chat",
) -> int:
    """Save Q&A pairs to a JSONL file.

    Args:
        pairs: Iterator of QAPair objects.
        output_path: Path to output JSONL file.
        system_prompt: System prompt to include in chat format.
        fmt: Output format — "chat" (messages list) or "raw" (question/answer fields).

    Returns:
        Number of pairs saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for pair in pairs:
            if fmt == "chat":
                record = {"messages": pair.to_chat_messages(system_prompt)}
            else:
                record = pair.to_dict()
            f.write(json.dumps(record) + "\n")
            count += 1

    logger.info("Saved %d Q&A pairs to %s", count, output_path)
    return count


def load_qa_pairs(path: str | Path) -> list[QAPair]:
    """Load Q&A pairs from a JSONL file (raw format)."""
    path = Path(path)
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "question" in data and "answer" in data:
                pairs.append(QAPair(
                    question=data["question"],
                    answer=data["answer"],
                    source_doc_id=data.get("source_doc_id", ""),
                    category=data.get("category", "qa"),
                    quality_score=data.get("quality_score", 1.0),
                    metadata=data.get("metadata", {}),
                ))
    return pairs
