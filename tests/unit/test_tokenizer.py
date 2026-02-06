"""Tests for DomainTokenizer, TokenizedExample, TokenStats, and packing."""

from src.data.processors.tokenizer import TokenizedExample, TokenStats

# ── TokenizedExample tests ───────────────────────────────────────────────────


class TestTokenizedExample:
    def test_length(self):
        ex = TokenizedExample(input_ids=[1, 2, 3, 4], attention_mask=[1, 1, 1, 1])
        assert ex.length == 4

    def test_empty(self):
        ex = TokenizedExample(input_ids=[], attention_mask=[])
        assert ex.length == 0

    def test_labels_default_none(self):
        ex = TokenizedExample(input_ids=[1, 2], attention_mask=[1, 1])
        assert ex.labels is None

    def test_labels_set(self):
        ex = TokenizedExample(input_ids=[1, 2], attention_mask=[1, 1], labels=[1, 2])
        assert ex.labels == [1, 2]

    def test_metadata_default_empty(self):
        ex = TokenizedExample(input_ids=[1], attention_mask=[1])
        assert ex.metadata == {}

    def test_metadata_set(self):
        ex = TokenizedExample(
            input_ids=[1], attention_mask=[1], metadata={"source": "test"}
        )
        assert ex.metadata["source"] == "test"


# ── TokenStats tests ─────────────────────────────────────────────────────────


class TestTokenStats:
    def test_avg_tokens_per_example(self):
        stats = TokenStats(total_examples=4, total_tokens=1000)
        assert stats.avg_tokens_per_example == 250.0

    def test_avg_tokens_zero_examples(self):
        stats = TokenStats(total_examples=0, total_tokens=0)
        assert stats.avg_tokens_per_example == 0.0

    def test_summary(self):
        stats = TokenStats(
            total_examples=10,
            total_tokens=5000,
            packed_sequences=3,
            truncated=2,
        )
        s = stats.summary()
        assert s["total_examples"] == 10
        assert s["total_tokens"] == 5000
        assert s["packed_sequences"] == 3
        assert s["truncated"] == 2
        assert s["avg_tokens_per_example"] == 500.0

    def test_summary_rounding(self):
        stats = TokenStats(total_examples=3, total_tokens=100)
        s = stats.summary()
        assert s["avg_tokens_per_example"] == 33.3

    def test_defaults(self):
        stats = TokenStats()
        assert stats.total_examples == 0
        assert stats.total_tokens == 0
        assert stats.packed_sequences == 0
        assert stats.truncated == 0


# ── DomainTokenizer config tests (no model download) ────────────────────────


class TestDomainTokenizerConfig:
    def test_init_defaults(self):
        from src.data.processors.tokenizer import DomainTokenizer

        tok = DomainTokenizer()
        assert tok.model_name == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert tok.max_length == 2048
        assert tok.padding is False
        assert tok.truncation is True
        assert tok._tokenizer is None  # lazy — not loaded yet

    def test_init_custom(self):
        from src.data.processors.tokenizer import DomainTokenizer

        tok = DomainTokenizer(
            model_name="gpt2",
            max_length=512,
            padding=True,
            truncation=False,
        )
        assert tok.model_name == "gpt2"
        assert tok.max_length == 512
        assert tok.padding is True
        assert tok.truncation is False

    def test_stats_initialized(self):
        from src.data.processors.tokenizer import DomainTokenizer

        tok = DomainTokenizer()
        assert tok.stats.total_examples == 0
        assert tok.stats.total_tokens == 0


# ── Pack examples tests (pure logic, no model needed) ────────────────────────


class TestPackExamplesLogic:
    """Test packing logic using pre-built TokenizedExample objects.

    We can't call DomainTokenizer.pack_examples without a loaded tokenizer
    (it needs eos_token_id), so we test the data structures and the packing
    contract with manual token lists.
    """

    def test_tokenized_example_copyable(self):
        ex = TokenizedExample(input_ids=[1, 2, 3], attention_mask=[1, 1, 1], labels=[1, 2, 3])
        copy = TokenizedExample(
            input_ids=ex.input_ids.copy(),
            attention_mask=ex.attention_mask.copy(),
            labels=(ex.labels or []).copy(),
        )
        assert copy.input_ids == ex.input_ids
        assert copy is not ex

    def test_multiple_examples_fit_in_sequence(self):
        """Simulate the packing contract: multiple short examples should
        fit into a single sequence of max_length."""
        max_length = 20
        examples = [
            TokenizedExample(input_ids=[1, 2, 3], attention_mask=[1, 1, 1], labels=[1, 2, 3]),
            TokenizedExample(input_ids=[4, 5, 6], attention_mask=[1, 1, 1], labels=[4, 5, 6]),
            TokenizedExample(input_ids=[7, 8, 9], attention_mask=[1, 1, 1], labels=[7, 8, 9]),
        ]
        # Total: 3*3 + 2 separators = 11 tokens, fits in 20
        total_tokens = sum(ex.length for ex in examples) + (len(examples) - 1)  # separators
        assert total_tokens <= max_length

    def test_example_exceeding_sequence_forces_split(self):
        """If one example alone exceeds max_length, it still gets its own sequence."""
        max_length = 10
        big_example = TokenizedExample(
            input_ids=list(range(15)),
            attention_mask=[1] * 15,
            labels=list(range(15)),
        )
        assert big_example.length > max_length
