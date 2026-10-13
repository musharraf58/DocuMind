import re

from app.services.extraction.base import CharBudget, ExtractionLimits, Extractor, Segment

# ATX headings: up to 3 spaces, 1-6 '#', a space, the title, optional closing '#'s.
_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _decode(data: bytes, limits: ExtractionLimits) -> str:
    # Uploads were already checked to be valid UTF-8; "replace" is a safety net.
    text = data.decode("utf-8", errors="replace")
    CharBudget(limits.max_chars).add(text)
    return text


class TextExtractor(Extractor):
    def extract(self, data: bytes, limits: ExtractionLimits) -> list[Segment]:
        return [Segment(_decode(data, limits))]


class MarkdownExtractor(Extractor):
    """Starts a new segment at each heading; the heading becomes its section title.

    Lines that look like headings inside fenced code blocks (``` or ~~~) are
    code, not headings (think "# comment" in a shell snippet).
    """

    def extract(self, data: bytes, limits: ExtractionLimits) -> list[Segment]:
        segments: list[Segment] = []
        title: str | None = None
        lines: list[str] = []
        fence: str | None = None

        for line in _decode(data, limits).splitlines():
            if fence is None:
                if opening := _FENCE.match(line):
                    fence = opening.group(1)
                elif heading := _HEADING.match(line):
                    segments.append(Segment("\n".join(lines), section_title=title))
                    title, lines = (heading.group(1) or "").strip() or None, []
                    continue
            elif line.lstrip().startswith(fence[0] * len(fence)):
                fence = None  # closing fence: same character, at least as long
            lines.append(line)

        segments.append(Segment("\n".join(lines), section_title=title))
        return segments
