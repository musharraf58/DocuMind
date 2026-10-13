import io

from pypdf import PdfReader

from app.services.extraction.base import (
    CharBudget,
    ExtractionError,
    ExtractionLimits,
    Extractor,
    Segment,
)


class PdfExtractor(Extractor):
    """One segment per page, numbered from 1.

    pypdf caps how much any single compressed stream may expand to (75 MB in
    pypdf 6), which protects against decompression bombs; the page and
    character limits bound the total work.
    """

    def extract(self, data: bytes, limits: ExtractionLimits) -> list[Segment]:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not self._decrypt_with_empty_password(reader):
                raise ExtractionError("This PDF is encrypted or password-protected")
            page_count = len(reader.pages)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError("File is not a readable PDF") from exc

        if page_count > limits.max_pages:
            raise ExtractionError(f"PDF has {page_count} pages; the limit is {limits.max_pages}")

        budget = CharBudget(limits.max_chars)
        segments = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                raise ExtractionError(f"Could not read the text on page {number}") from exc
            budget.add(text)
            segments.append(Segment(text, page_number=number))
        return segments

    @staticmethod
    def _decrypt_with_empty_password(reader: PdfReader) -> bool:
        # Many PDFs are "encrypted" only to restrict printing or copying, with
        # an empty password for opening; those read fine.
        try:
            return bool(reader.decrypt(""))
        except Exception:
            return False  # e.g. AES encryption needs the optional cryptography package
