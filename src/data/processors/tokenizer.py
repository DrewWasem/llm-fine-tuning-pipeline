"""Tokenizer wrapper with packing support for efficient training.

Wraps HuggingFace tokenizers with utilities for:
- Applying chat templates
- Packing multiple short examples into a single sequence
- Tracking token statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.data.formatters.domain_chat import ChatExample

logger = logging.getLogger(__name__)


@dataclass
class TokenizedExample:
    """A tokenized training example."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int] | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.input_ids)


@dataclass
class TokenStats:
    """Statistics about tokenization."""

    total_examples: int = 0
    total_tokens: int = 0
    packed_sequences: int = 0
    truncated: int = 0

    @property
    def avg_tokens_per_example(self) -> float:
        if self.total_examples == 0:
            return 0.0
        return self.total_tokens / self.total_examples

    def summary(self) -> dict:
        return {
            "total_examples": self.total_examples,
            "total_tokens": self.total_tokens,
            "packed_sequences": self.packed_sequences,
            "truncated": self.truncated,
            "avg_tokens_per_example": round(self.avg_tokens_per_example, 1),
        }


class DomainTokenizer:
    """Tokenizer wrapper for domain adaptation training.

    Args:
        model_name: HuggingFace model name or path for the tokenizer.
        max_length: Maximum sequence length.
        padding: Whether to pad sequences to max_length.
        truncation: Whether to truncate sequences exceeding max_length.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_length: int = 2048,
        padding: bool = False,
        truncation: bool = True,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.stats = TokenStats()
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer on first access."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info("Loaded tokenizer: %s (vocab=%d)", self.model_name, len(self._tokenizer))
        return self._tokenizer

    def tokenize_chat(self, example: ChatExample) -> TokenizedExample:
        """Tokenize a chat example using the model's chat template."""
        messages = [{"role": m.role, "content": m.content} for m in example.messages]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding="max_length" if self.padding else False,
            return_tensors=None,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        self.stats.total_examples += 1
        self.stats.total_tokens += len(input_ids)
        if len(input_ids) == self.max_length:
            self.stats.truncated += 1

        return TokenizedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids.copy(),
            metadata=example.metadata or {},
        )

    def pack_examples(
        self, examples: list[TokenizedExample], separator_id: int | None = None
    ) -> list[TokenizedExample]:
        """Pack multiple short examples into single max_length sequences.

        This improves training efficiency by reducing padding waste.

        Args:
            examples: List of tokenized examples to pack.
            separator_id: Token ID to insert between packed examples.
                         Defaults to the EOS token.
        """
        if separator_id is None:
            separator_id = self.tokenizer.eos_token_id

        packed: list[TokenizedExample] = []
        current_ids: list[int] = []
        current_mask: list[int] = []
        current_labels: list[int] = []

        for ex in examples:
            # Check if adding this example would exceed max_length
            needed = len(ex.input_ids) + (1 if current_ids else 0)  # +1 for separator
            if current_ids and len(current_ids) + needed > self.max_length:
                # Finalize current packed sequence
                packed.append(TokenizedExample(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    labels=current_labels,
                ))
                self.stats.packed_sequences += 1
                current_ids = []
                current_mask = []
                current_labels = []

            # Add separator between examples
            if current_ids:
                current_ids.append(separator_id)
                current_mask.append(1)
                current_labels.append(-100)  # Don't compute loss on separator

            current_ids.extend(ex.input_ids)
            current_mask.extend(ex.attention_mask)
            current_labels.extend(ex.labels or ex.input_ids)

        # Don't forget the last sequence
        if current_ids:
            packed.append(TokenizedExample(
                input_ids=current_ids,
                attention_mask=current_mask,
                labels=current_labels,
            ))
            self.stats.packed_sequences += 1

        logger.info(
            "Packed %d examples into %d sequences (avg %.0f tokens/seq)",
            len(examples),
            len(packed),
            sum(p.length for p in packed) / max(len(packed), 1),
        )
        return packed
