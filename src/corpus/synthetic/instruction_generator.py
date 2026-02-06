"""Generate instruction-response pairs for domain SFT training.

Creates diverse instruction data by combining task templates
(summarize, analyze, explain, classify, etc.) with document passages.
Supports template-based and LLM-based generation.
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

# Task templates per domain.  Each entry is (task_type, instruction_template).
# Templates with {passage} are filled with document text.
# Templates with {term} are filled with terminology terms.
LEGAL_INSTRUCTION_TEMPLATES: list[tuple[str, str]] = [
    ("summarize", "Summarize the following legal text in plain language.\n\n{passage}"),
    ("summarize", "Provide a brief summary of this legal document.\n\n{passage}"),
    ("summarize", "Write a concise summary of the key points in this legal passage.\n\n{passage}"),
    ("analyze", "Analyze the legal implications of the following clause.\n\n{passage}"),
    ("analyze", "What are the key legal issues raised in this passage?\n\n{passage}"),
    ("analyze", "Identify the legal rights and obligations described in this text.\n\n{passage}"),
    ("explain", "Explain the following legal concept to a non-lawyer.\n\n{passage}"),
    ("explain", "What does this legal provision mean in practical terms?\n\n{passage}"),
    ("classify", "Classify the following legal text into its relevant area of law.\n\n{passage}"),
    ("classify", "What type of legal document is this? Explain your reasoning.\n\n{passage}"),
    ("extract", "Extract the key legal terms and definitions from this text.\n\n{passage}"),
    ("extract", "List the parties, obligations, and conditions described in this clause.\n\n{passage}"),
    ("compare", "Compare the legal positions described in this passage and identify any conflicts.\n\n{passage}"),
    ("draft", "Rewrite the following legal clause in simpler language while preserving its meaning.\n\n{passage}"),
]

MEDICAL_INSTRUCTION_TEMPLATES: list[tuple[str, str]] = [
    ("summarize", "Summarize the following clinical findings.\n\n{passage}"),
    ("explain", "Explain this medical concept for a patient.\n\n{passage}"),
    ("analyze", "What are the key diagnostic indicators in this passage?\n\n{passage}"),
    ("classify", "Classify the condition described in this text.\n\n{passage}"),
]

FINANCIAL_INSTRUCTION_TEMPLATES: list[tuple[str, str]] = [
    ("summarize", "Summarize the key financial metrics from this report.\n\n{passage}"),
    ("analyze", "Analyze the investment risks described in this passage.\n\n{passage}"),
    ("explain", "Explain this financial concept in simple terms.\n\n{passage}"),
    ("classify", "Classify the type of financial instrument described.\n\n{passage}"),
]

TERM_DEFINITION_TEMPLATES: list[str] = [
    "Define the legal term '{term}' and provide an example of its use.",
    "What does '{term}' mean in a legal context?",
    "Explain the concept of '{term}' as it applies in law.",
    "Provide a clear definition of '{term}' and explain when it typically arises.",
]

DOMAIN_INSTRUCTION_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "legal": LEGAL_INSTRUCTION_TEMPLATES,
    "medical": MEDICAL_INSTRUCTION_TEMPLATES,
    "financial": FINANCIAL_INSTRUCTION_TEMPLATES,
}


@dataclass
class InstructionExample:
    """A single instruction-response pair for SFT training."""

    instruction: str
    response: str
    task_type: str = "general"
    source_doc_id: str = ""
    quality_score: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_chat_messages(self, system_prompt: str = "") -> list[dict]:
        """Convert to chat-message format for training."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.instruction})
        messages.append({"role": "assistant", "content": self.response})
        return messages

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "response": self.response,
            "task_type": self.task_type,
            "source_doc_id": self.source_doc_id,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }


def _split_into_passages(text: str, max_length: int = 1500) -> list[str]:
    """Split text into passages for instruction generation."""
    if len(text) <= max_length:
        return [text.strip()] if text.strip() else []

    passages = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        if end < len(text):
            # Break at paragraph or sentence boundary
            for delim in ("\n\n", ".\n", ". ", "\n"):
                pos = text.rfind(delim, start + max_length // 2, end)
                if pos > start:
                    end = pos + len(delim)
                    break
        chunk = text[start:end].strip()
        if chunk and len(chunk.split()) >= 20:
            passages.append(chunk)
        start = end

    return passages


class InstructionGenerator:
    """Generate instruction-response pairs from domain documents.

    Args:
        domain: Domain name for template selection.
        templates: Custom instruction templates as (task_type, template) tuples.
        passage_length: Target character length for document passages.
        min_doc_length: Minimum document length to process.
        max_examples_per_doc: Maximum instruction pairs per document.
        terminology: List of domain terms for definition generation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        domain: str = "legal",
        templates: list[tuple[str, str]] | None = None,
        passage_length: int = 1500,
        min_doc_length: int = 200,
        max_examples_per_doc: int = 3,
        terminology: list[str] | None = None,
        seed: int = 42,
    ):
        self.domain = domain
        self.templates = templates or DOMAIN_INSTRUCTION_TEMPLATES.get(
            domain, LEGAL_INSTRUCTION_TEMPLATES
        )
        self.passage_length = passage_length
        self.min_doc_length = min_doc_length
        self.max_examples_per_doc = max_examples_per_doc
        self.terminology = terminology or []
        self.rng = random.Random(seed)

    def generate_from_document(self, doc: Document) -> list[InstructionExample]:
        """Generate instruction-response pairs from a single document.

        Randomly selects passages and pairs them with task templates.
        The document passage is used as both context and reference answer.
        """
        if len(doc.text) < self.min_doc_length:
            return []

        passages = _split_into_passages(doc.text, max_length=self.passage_length)
        if not passages:
            return []

        examples = []
        n_passages = min(len(passages), self.max_examples_per_doc)
        selected_passages = self.rng.sample(passages, n_passages)

        for passage in selected_passages:
            task_type, template = self.rng.choice(self.templates)
            instruction = template.format(passage=passage, term="")

            examples.append(InstructionExample(
                instruction=instruction,
                response=passage,
                task_type=task_type,
                source_doc_id=doc.doc_id,
                metadata={"domain": self.domain, "template_type": task_type},
            ))

        return examples

    def generate_term_definitions(
        self, max_terms: int | None = None
    ) -> list[InstructionExample]:
        """Generate definition-style instruction pairs from terminology list.

        Each term gets a template-based instruction asking for its definition.
        The response is a placeholder that should be filled by an LLM or human.
        """
        if not self.terminology:
            return []

        terms = self.terminology
        if max_terms and max_terms < len(terms):
            terms = self.rng.sample(terms, max_terms)

        examples = []
        for term in terms:
            template = self.rng.choice(TERM_DEFINITION_TEMPLATES)
            instruction = template.format(term=term)

            examples.append(InstructionExample(
                instruction=instruction,
                response=f"[Definition of '{term}' — to be generated by LLM or domain expert]",
                task_type="definition",
                metadata={"domain": self.domain, "term": term},
            ))

        return examples

    def generate_term_definitions_with_model(
        self,
        model,
        tokenizer,
        max_terms: int | None = None,
        max_new_tokens: int = 256,
    ) -> list[InstructionExample]:
        """Generate term definitions using an LLM to produce responses."""
        from src.evaluation.base import generate_text

        if not self.terminology:
            return []

        terms = self.terminology
        if max_terms and max_terms < len(terms):
            terms = self.rng.sample(terms, max_terms)

        examples = []
        for term in terms:
            template = self.rng.choice(TERM_DEFINITION_TEMPLATES)
            instruction = template.format(term=term)

            response = generate_text(model, tokenizer, instruction, max_new_tokens=max_new_tokens)
            if response and len(response) > 20:
                examples.append(InstructionExample(
                    instruction=instruction,
                    response=response,
                    task_type="definition",
                    metadata={
                        "domain": self.domain,
                        "term": term,
                        "generation_method": "llm",
                    },
                ))

        return examples

    def generate_from_documents(
        self, documents: Iterator[Document], max_examples: int | None = None
    ) -> Iterator[InstructionExample]:
        """Generate instruction pairs from a stream of documents."""
        count = 0
        for doc in documents:
            if max_examples and count >= max_examples:
                break
            examples = self.generate_from_document(doc)
            for ex in examples:
                if max_examples and count >= max_examples:
                    break
                yield ex
                count += 1

        logger.info("Generated %d instruction examples", count)

    def generate_with_model(
        self, doc: Document, model, tokenizer, max_new_tokens: int = 512
    ) -> list[InstructionExample]:
        """Use an LLM to generate responses for instruction-document pairs."""
        from src.evaluation.base import generate_text

        if len(doc.text) < self.min_doc_length:
            return []

        passages = _split_into_passages(doc.text, max_length=self.passage_length)
        if not passages:
            return []

        examples = []
        n_passages = min(len(passages), self.max_examples_per_doc)
        selected_passages = self.rng.sample(passages, n_passages)

        for passage in selected_passages:
            task_type, template = self.rng.choice(self.templates)
            instruction = template.format(passage=passage[:800], term="")

            response = generate_text(model, tokenizer, instruction, max_new_tokens=max_new_tokens)
            if response and len(response) > 20:
                examples.append(InstructionExample(
                    instruction=instruction,
                    response=response,
                    task_type=task_type,
                    source_doc_id=doc.doc_id,
                    metadata={
                        "domain": self.domain,
                        "template_type": task_type,
                        "generation_method": "llm",
                    },
                ))

        return examples


def save_instructions(
    examples: Iterator[InstructionExample],
    output_path: str | Path,
    system_prompt: str = "",
    fmt: str = "chat",
) -> int:
    """Save instruction examples to a JSONL file.

    Args:
        examples: Iterator of InstructionExample objects.
        output_path: Path to output JSONL file.
        system_prompt: System prompt to include in chat format.
        fmt: Output format — "chat" (messages list) or "raw" (instruction/response).

    Returns:
        Number of examples saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for ex in examples:
            if fmt == "chat":
                record = {"messages": ex.to_chat_messages(system_prompt)}
            else:
                record = ex.to_dict()
            f.write(json.dumps(record) + "\n")
            count += 1

    logger.info("Saved %d instruction examples to %s", count, output_path)
    return count
