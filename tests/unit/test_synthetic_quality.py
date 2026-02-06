"""Tests for synthetic data quality filtering."""

from pytest import approx

from src.corpus.synthetic.instruction_generator import InstructionExample
from src.corpus.synthetic.qa_generator import QAPair
from src.corpus.synthetic.quality_filter import (
    FilterStats,
    SyntheticQualityFilter,
    _quality_score,
    _repetition_ratio,
    _sentence_repetition_ratio,
)

# ── Repetition ratio tests ──────────────────────────────────────────────────


class TestRepetitionRatio:
    def test_no_repetition(self):
        text = "The court held that the defendant was liable for breach of contract."
        ratio = _repetition_ratio(text)
        assert ratio == 0.0

    def test_high_repetition(self):
        text = "the same words repeated " * 20
        ratio = _repetition_ratio(text)
        assert ratio > 0.5

    def test_short_text(self):
        assert _repetition_ratio("hello world") == 0.0

    def test_empty_text(self):
        assert _repetition_ratio("") == 0.0


class TestSentenceRepetitionRatio:
    def test_no_repetition(self):
        text = "First sentence. Second sentence. Third sentence."
        assert _sentence_repetition_ratio(text) == 0.0

    def test_all_repeated(self):
        text = "Same sentence. Same sentence. Same sentence."
        ratio = _sentence_repetition_ratio(text)
        assert ratio > 0.5

    def test_single_sentence(self):
        assert _sentence_repetition_ratio("Just one.") == 0.0

    def test_empty(self):
        assert _sentence_repetition_ratio("") == 0.0


# ── Quality score tests ──────────────────────────────────────────────────────


class TestQualityScore:
    def test_good_text(self):
        text = (
            "The court ruled that the defendant breached the contract. "
            "This was based on the failure to perform contractual obligations. "
            "The plaintiff was awarded damages as a result."
        )
        score = _quality_score(text)
        assert score >= 0.7

    def test_empty_text(self):
        assert _quality_score("") == 0.0
        assert _quality_score("   ") == 0.0

    def test_mostly_numbers(self):
        text = "123 456 789 0 " * 30
        score = _quality_score(text)
        assert score < 0.7

    def test_repetitive_text(self):
        text = "the same thing over and over. " * 50
        score = _quality_score(text)
        assert score < 0.8

    def test_excessive_whitespace(self):
        text = "word " + "   " * 100 + "word"
        score = _quality_score(text)
        assert score < 1.0


# ── FilterStats tests ────────────────────────────────────────────────────────


class TestFilterStats:
    def test_pass_rate(self):
        stats = FilterStats(total=10, passed=7)
        assert stats.pass_rate == approx(0.7)

    def test_pass_rate_zero(self):
        stats = FilterStats(total=0, passed=0)
        assert stats.pass_rate == 0.0

    def test_to_dict(self):
        stats = FilterStats(total=100, passed=80, rejected_too_short=10, rejected_repetition=10)
        d = stats.to_dict()
        assert d["total"] == 100
        assert d["passed"] == 80
        assert d["pass_rate"] == approx(0.8)


# ── SyntheticQualityFilter tests ─────────────────────────────────────────────


class TestSyntheticQualityFilter:
    def test_pass_good_text(self):
        filt = SyntheticQualityFilter(min_length=10, min_quality_score=0.3)
        text = (
            "The court ruled that the defendant was liable for the breach. "
            "The plaintiff was awarded compensatory damages in the amount of ten thousand dollars."
        )
        ok, reason = filt.check(text)
        assert ok is True
        assert reason == "passed"

    def test_reject_too_short(self):
        filt = SyntheticQualityFilter(min_length=100)
        ok, reason = filt.check("Too short.")
        assert ok is False
        assert reason == "too_short"

    def test_reject_too_long(self):
        filt = SyntheticQualityFilter(max_length=100)
        ok, reason = filt.check("x " * 200)
        assert ok is False
        assert reason == "too_long"

    def test_reject_repetitive(self):
        filt = SyntheticQualityFilter(min_length=10, max_repetition_ratio=0.3)
        text = "repeat these words " * 50
        ok, reason = filt.check(text)
        assert ok is False
        assert reason == "too_repetitive"

    def test_stats_tracking(self):
        filt = SyntheticQualityFilter(min_length=10, max_length=500, min_quality_score=0.3)
        filt.check(
            "The court ruled that the defendant breached the contract. "
            "This was based on the failure to perform obligations. "
            "The plaintiff was awarded damages as a result of the breach."
        )  # pass
        filt.check("short")  # fail: too short
        filt.check("x " * 1000)  # fail: too long

        assert filt.stats.total == 3
        assert filt.stats.passed == 1
        assert filt.stats.rejected_too_short == 1
        assert filt.stats.rejected_too_long == 1

    def test_reset_stats(self):
        filt = SyntheticQualityFilter()
        filt.check("text " * 20)
        filt.reset_stats()
        assert filt.stats.total == 0

    def test_filter_qa_pairs(self):
        filt = SyntheticQualityFilter(min_length=10, min_quality_score=0.3)
        good_answer = (
            "The court held that the defendant was liable for breach of contract. "
            "This ruling was based on the failure to deliver goods on time. "
            "The plaintiff successfully proved all elements of the claim."
        )
        pairs = [
            QAPair(question="Q?", answer=good_answer),
            QAPair(question="Q?", answer="Bad."),  # too short
        ]
        passed = filt.filter_qa_pairs(pairs, field="answer")
        assert len(passed) == 1
        assert passed[0].answer.startswith("The court held")

    def test_filter_instruction_examples(self):
        filt = SyntheticQualityFilter(min_length=10, min_quality_score=0.3)
        good_response = (
            "The doctrine of respondeat superior holds an employer liable for employee actions. "
            "In this case, the court examined whether the employee was acting within scope. "
            "The verdict was in favor of the plaintiff based on established precedent."
        )
        examples = [
            InstructionExample(instruction="Summarize.", response=good_response),
            InstructionExample(instruction="Explain.", response="X."),  # too short
        ]
        passed = filt.filter_qa_pairs(examples, field="response")
        assert len(passed) == 1

    def test_quality_score_updated(self):
        filt = SyntheticQualityFilter(min_length=10, min_quality_score=0.0)
        good_text = (
            "The Supreme Court reversed the lower court decision in a landmark ruling. "
            "Justice Roberts delivered the majority opinion addressing constitutional issues. "
            "The dissenting justices argued that the precedent should have been upheld."
        )
        pair = QAPair(
            question="Q?",
            answer=good_text,
            quality_score=0.0,
        )
        passed = filt.filter_qa_pairs([pair], field="answer")
        assert len(passed) == 1
        assert passed[0].quality_score > 0.0  # Should have been updated
