"""The evaluation question file: loading, validation, and passage matching."""

import textwrap

import pytest

from eval.dataset import (
    DatasetError,
    Expected,
    is_expected_chunk,
    load_questions,
)

DOC = "# Refunds\n\nThe refund is paid within 14 days\nto the original payment method.\n"


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "refunds.md").write_text(DOC)
    (folder / "notes.bin").write_bytes(b"ignored: not a supported type")
    return folder


def load(tmp_path, corpus, body: str):
    path = tmp_path / "questions.yaml"
    path.write_text(textwrap.dedent(body))
    return load_questions(path, corpus)


VALID = """
questions:
  - id: refund-timing
    type: paraphrase
    question: When do I get my money back?
    expected:
      document: refunds.md
      must_contain: "REFUND is paid   within 14 days to the original"
  - id: off-topic
    type: unanswerable
    question: How do I bake bread?
"""


def test_valid_file_loads(tmp_path, corpus):
    answerable, unanswerable = load(tmp_path, corpus, VALID)
    assert answerable.type == "paraphrase" and answerable.answerable
    assert answerable.expected.document == "refunds.md"
    assert unanswerable.expected is None and not unanswerable.answerable


def test_the_real_question_set_is_valid():
    questions = load_questions()
    assert len(questions) >= 12
    assert {q.type for q in questions} == {
        "answerable",
        "keyword",
        "paraphrase",
        "short",
        "unanswerable",
    }


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("questions: []", "no questions"),
        ("hello", "'questions:' list"),
        ("questions: [unclosed", "not valid YAML"),
        (
            """
            questions:
              - {id: Bad_Id, type: short, question: q}
            """,
            "lowercase letters",
        ),
        (
            """
            questions:
              - {id: q1, type: trivia, question: q}
            """,
            "'type' must be one of",
        ),
        (
            """
            questions:
              - {id: q1, type: short, question: "  "}
            """,
            "non-empty",
        ),
        (
            """
            questions:
              - {id: q1, type: short, question: refunds}
            """,
            "needs 'expected'",
        ),
        (
            """
            questions:
              - id: q1
                type: unanswerable
                question: bread?
                expected: {document: refunds.md, must_contain: refund}
            """,
            "unanswerable question has no 'expected'",
        ),
        (
            """
            questions:
              - id: q1
                type: short
                question: refunds
                expected: {document: missing.md, must_contain: refund}
            """,
            "not in eval/corpus/",
        ),
        (
            """
            questions:
              - id: q1
                type: short
                question: refunds
                expected: {document: refunds.md, must_contain: paid within 41 days}
            """,
            "does not appear in refunds.md",
        ),
        (
            """
            questions:
              - id: q1
                type: short
                question: refunds
                expected: {document: refunds.md, must_contain: refund, page: 2}
            """,
            "does not appear in refunds.md on page 2",
        ),
        (
            """
            questions:
              - id: q1
                type: short
                question: refunds
                expected: {document: refunds.md, must_contain: refund, page: 0}
            """,
            "page number",
        ),
        (
            """
            questions:
              - id: q1
                type: short
                question: refunds
                expected: {document: refunds.md, must_contain: refund, chunk_id: 7}
            """,
            "unknown field(s) chunk_id",
        ),
        (
            """
            questions:
              - {id: q1, type: unanswerable, question: "a?"}
              - {id: q1, type: unanswerable, question: "b?"}
            """,
            "used twice",
        ),
    ],
)
def test_invalid_files_are_rejected_with_a_useful_message(tmp_path, corpus, body, message):
    with pytest.raises(DatasetError, match=message.replace("(", r"\(").replace(")", r"\)")):
        load(tmp_path, corpus, body)


def test_over_long_passage_is_rejected(tmp_path, corpus):
    body = f"""
    questions:
      - id: q1
        type: short
        question: refunds
        expected: {{document: refunds.md, must_contain: "{"refund " * 50}"}}
    """
    with pytest.raises(DatasetError, match="over 200 characters"):
        load(tmp_path, corpus, body)


EXPECTED = Expected("refunds.md", "paid within 14 days", None)


@pytest.mark.parametrize(
    ("document", "content", "page", "expected", "matches"),
    [
        ("refunds.md", "The refund is PAID within\n14 days.", None, EXPECTED, True),
        ("other.md", "The refund is paid within 14 days.", None, EXPECTED, False),
        ("refunds.md", "The refund is paid within 30 days.", None, EXPECTED, False),
        ("refunds.md", "paid within 14 days", 3, Expected("refunds.md", "14 days", 3), True),
        ("refunds.md", "paid within 14 days", 4, Expected("refunds.md", "14 days", 3), False),
    ],
)
def test_expected_chunk_matching(document, content, page, expected, matches):
    assert is_expected_chunk(expected, document, content, page) is matches
