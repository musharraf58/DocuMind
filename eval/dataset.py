"""The evaluation set: loading, validating, and deciding which chunk is "right".

A question names the passage that answers it (document file name, a short
text the right chunk must contain, optionally the page), never a chunk id.
Chunk ids change every time a document is re-processed or the chunk size
changes, so a set written against ids would silently break; a passage
survives re-chunking.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from app.core.config import get_settings
from app.services.documents import ALLOWED_TYPES
from app.services.extraction import ExtractionLimits, Segment, extract

EVAL_DIR = Path(__file__).parent
QUESTIONS_FILE = EVAL_DIR / "questions.yaml"
CORPUS_DIR = EVAL_DIR / "corpus"

QuestionType = Literal["answerable", "keyword", "paraphrase", "short", "unanswerable"]
QUESTION_TYPES: tuple[str, ...] = get_args(QuestionType)
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
# A passage must fit comfortably inside the smallest chunk we try (300
# tokens); longer passages would often be split across two chunks.
MAX_PASSAGE_CHARS = 200


class DatasetError(ValueError):
    """The question file is malformed. The message says where and why."""


@dataclass(frozen=True)
class Expected:
    document: str
    must_contain: str
    page: int | None = None


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    type: QuestionType
    expected: Expected | None  # None exactly for unanswerable questions

    @property
    def answerable(self) -> bool:
        return self.expected is not None


def normalize(text: str) -> str:
    """Case and whitespace don't matter when matching a passage: chunking and
    extraction re-flow line breaks."""
    return " ".join(text.split()).casefold()


def is_expected_chunk(
    expected: Expected, document_name: str, content: str, page: int | None
) -> bool:
    """Is this retrieved chunk the one that answers the question?"""
    if document_name != expected.document:
        return False
    if expected.page is not None and page != expected.page:
        return False
    return normalize(expected.must_contain) in normalize(content)


def corpus_files(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    """The documents to ingest: every supported file in the corpus folder."""
    return sorted(
        path
        for path in corpus_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_TYPES
    )


def load_questions(path: Path = QUESTIONS_FILE, corpus_dir: Path = CORPUS_DIR) -> list[Question]:
    """Load and fully validate the question set. Raises DatasetError naming the
    question and the problem, so a typo fails here rather than showing up as
    a mysterious "miss" in the results."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
        raise DatasetError(f"{path.name} must contain a 'questions:' list")
    if not raw["questions"]:
        raise DatasetError(f"{path.name} has no questions")

    documents = {file.name: _segments(file) for file in corpus_files(corpus_dir)}
    questions: list[Question] = []
    seen: set[str] = set()
    for position, item in enumerate(raw["questions"], start=1):
        question = _parse_question(item, position, documents)
        if question.id in seen:
            raise DatasetError(f"question id '{question.id}' is used twice")
        seen.add(question.id)
        questions.append(question)
    return questions


def _parse_question(item: Any, position: int, documents: dict[str, list[Segment]]) -> Question:
    where = f"question #{position}"
    if not isinstance(item, dict):
        raise DatasetError(f"{where}: must be a mapping with id, question, type")
    _no_unknown_keys(item, {"id", "question", "type", "expected"}, where)

    question_id = item.get("id")
    if not isinstance(question_id, str) or not ID_PATTERN.match(question_id):
        raise DatasetError(f"{where}: 'id' must be lowercase letters, digits and dashes")
    where = f"question '{question_id}'"

    text = item.get("question")
    if not isinstance(text, str) or not text.strip():
        raise DatasetError(f"{where}: 'question' must be non-empty text")
    kind = item.get("type")
    if kind not in QUESTION_TYPES:
        raise DatasetError(f"{where}: 'type' must be one of {', '.join(QUESTION_TYPES)}")

    if kind == "unanswerable":
        if "expected" in item:
            raise DatasetError(f"{where}: an unanswerable question has no 'expected'")
        return Question(question_id, text.strip(), kind, None)
    if "expected" not in item:
        raise DatasetError(f"{where}: needs 'expected' (document and must_contain)")
    expected = _parse_expected(item["expected"], where, documents)
    return Question(question_id, text.strip(), kind, expected)


def _parse_expected(raw: Any, where: str, documents: dict[str, list[Segment]]) -> Expected:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: 'expected' must be a mapping")
    _no_unknown_keys(raw, {"document", "must_contain", "page"}, f"{where} expected")

    document = raw.get("document")
    if document not in documents:
        raise DatasetError(
            f"{where}: document {document!r} is not in eval/corpus/ "
            f"(found: {', '.join(sorted(documents))})"
        )
    passage = raw.get("must_contain")
    if not isinstance(passage, str) or not passage.strip():
        raise DatasetError(f"{where}: 'must_contain' must be non-empty text")
    if len(passage) > MAX_PASSAGE_CHARS:
        raise DatasetError(
            f"{where}: 'must_contain' is over {MAX_PASSAGE_CHARS} characters; use a "
            "shorter, distinctive phrase so it can't be split between two chunks"
        )
    page = raw.get("page")
    if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
        raise DatasetError(f"{where}: 'page' must be a page number (1 or more)")

    segments = [s for s in documents[document] if page is None or s.page_number == page]
    if not any(normalize(passage) in normalize(s.text) for s in segments):
        on_page = f" on page {page}" if page is not None else ""
        raise DatasetError(f"{where}: {passage!r} does not appear in {document}{on_page}")
    return Expected(document, passage, page)


def _no_unknown_keys(mapping: dict, allowed: set[str], where: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise DatasetError(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")


def _segments(file: Path) -> list[Segment]:
    """The document's text exactly as the app extracts it, so the passage check
    sees what the chunks will contain."""
    limits = ExtractionLimits.from_settings(get_settings())
    return extract(file.read_bytes(), ALLOWED_TYPES[file.suffix.lower()], limits)


if __name__ == "__main__":
    from collections import Counter

    try:
        loaded = load_questions()
    except DatasetError as exc:
        raise SystemExit(f"questions.yaml: {exc}") from None
    counts = Counter(q.type for q in loaded)
    print(f"{len(loaded)} questions OK over {len(corpus_files())} documents:")
    for kind in QUESTION_TYPES:
        print(f"  {kind:<13} {counts[kind]}")
