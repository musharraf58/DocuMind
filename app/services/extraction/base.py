from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import Settings


@dataclass(frozen=True)
class Segment:
    """A run of text from one place in a document: one PDF page, or one section
    under a heading. Chunks never span two segments, so each chunk keeps a
    single page number and section title."""

    text: str
    page_number: int | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class ExtractionLimits:
    max_pages: int
    max_chars: int
    min_chars: int
    max_docx_uncompressed_bytes: int
    max_docx_entries: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "ExtractionLimits":
        return cls(
            max_pages=settings.MAX_EXTRACT_PAGES,
            max_chars=settings.MAX_EXTRACTED_CHARS,
            min_chars=settings.MIN_EXTRACTED_CHARS,
            max_docx_uncompressed_bytes=settings.MAX_DOCX_UNCOMPRESSED_MB * 1024 * 1024,
            max_docx_entries=settings.MAX_DOCX_ENTRIES,
        )


class ExtractionError(Exception):
    """The file can't be turned into text. The message is shown to the user, so
    it must be short and must not contain internals (paths, stack traces)."""


class CharBudget:
    """Counts extracted characters so a huge document is rejected as soon as it
    goes over the limit, instead of after everything is in memory."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.used = 0

    def add(self, text: str) -> None:
        self.used += len(text)
        if self.used > self.max_chars:
            raise ExtractionError(f"Document has more than {self.max_chars:,} characters of text")


class Extractor(ABC):
    """Turns one file type into segments. Implementations are synchronous and
    CPU-bound; callers run them in a worker thread."""

    @abstractmethod
    def extract(self, data: bytes, limits: ExtractionLimits) -> list[Segment]:
        """Return raw (uncleaned) segments in document order.

        Raise ExtractionError for problems with the file. Any other exception
        is treated as an unreadable file by the caller.
        """
