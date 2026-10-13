"""Turn uploaded files into clean text segments.

Usage: extract(data, content_type, limits) -> list[Segment]. Runs synchronously
and can be slow on big files, so call it from a worker thread.
"""

from app.services.extraction.base import ExtractionError, ExtractionLimits, Extractor, Segment
from app.services.extraction.cleaning import clean_text
from app.services.extraction.text import MarkdownExtractor, TextExtractor

__all__ = ["ExtractionError", "ExtractionLimits", "Segment", "extract"]

MAX_TITLE_LENGTH = 300

# Keyed by the content type recorded at upload, which we set from the
# extension and verified against the bytes; never the client's claim.
EXTRACTORS: dict[str, Extractor] = {
    "text/markdown": MarkdownExtractor(),
    "text/plain": TextExtractor(),
}


def _clean_title(title: str | None) -> str | None:
    if title is None:
        return None
    title = " ".join(clean_text(title).split())[:MAX_TITLE_LENGTH]
    return title or None


def extract(data: bytes, content_type: str, limits: ExtractionLimits) -> list[Segment]:
    """Extract, clean, and drop empty segments.

    Raises ExtractionError (with a user-safe message) when the file is over a
    limit or yields almost no text.
    """
    extractor = EXTRACTORS.get(content_type)
    if extractor is None:
        raise ExtractionError("This file type can't be processed")

    segments = []
    for raw in extractor.extract(data, limits):
        text = clean_text(raw.text)
        if text:
            segments.append(Segment(text, raw.page_number, _clean_title(raw.section_title)))

    visible_chars = sum(len("".join(s.text.split())) for s in segments)
    if visible_chars < limits.min_chars:
        raise ExtractionError(
            "No readable text found. If this is a scanned document, it needs OCR, "
            "which isn't supported yet."
        )
    return segments
