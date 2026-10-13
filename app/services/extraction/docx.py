import io
import zipfile

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.services.extraction.base import (
    CharBudget,
    ExtractionError,
    ExtractionLimits,
    Extractor,
    Segment,
)

HEADING_STYLES = ("Title", "Heading")


def check_zip_limits(data: bytes, limits: ExtractionLimits) -> None:
    """Refuse zip bombs: small archives that expand to enormous sizes.

    Python's zipfile never decompresses an entry past the size its header
    declares, so the sum of declared sizes really does cap the memory that
    reading the archive can use, even when the header lies.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise ExtractionError("File is not a readable DOCX") from exc

    if len(entries) > limits.max_docx_entries:
        raise ExtractionError("DOCX file contains too many parts")
    if sum(entry.file_size for entry in entries) > limits.max_docx_uncompressed_bytes:
        mb = limits.max_docx_uncompressed_bytes // (1024 * 1024)
        raise ExtractionError(f"DOCX file expands to more than {mb} MB")


def _is_heading(paragraph: Paragraph) -> bool:
    style = paragraph.style
    return style is not None and (style.name or "").startswith(HEADING_STYLES)


def _table_text(table: Table) -> str:
    """One line per row, cells separated by " | "."""
    rows = []
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:
            # A cell merged across columns is returned once per column it spans.
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(" ".join(cell.text.split()))
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


class DocxExtractor(Extractor):
    """Body paragraphs and tables in document order; each heading starts a new
    segment. Word files have no fixed pages, so page_number is always None.
    Headers, footers, footnotes and comments are not extracted."""

    def extract(self, data: bytes, limits: ExtractionLimits) -> list[Segment]:
        check_zip_limits(data, limits)
        try:
            document = docx.Document(io.BytesIO(data))
            blocks = list(document.iter_inner_content())
        except Exception as exc:
            raise ExtractionError("File is not a readable DOCX") from exc

        budget = CharBudget(limits.max_chars)
        segments: list[Segment] = []
        title: str | None = None
        parts: list[str] = []
        for block in blocks:
            text = _table_text(block) if isinstance(block, Table) else block.text
            budget.add(text)
            if isinstance(block, Paragraph) and _is_heading(block):
                segments.append(Segment("\n\n".join(parts), section_title=title))
                title, parts = text, []
            else:
                # Blank lines between blocks, so the chunker sees paragraph breaks.
                parts.append(text)
        segments.append(Segment("\n\n".join(parts), section_title=title))
        return segments
